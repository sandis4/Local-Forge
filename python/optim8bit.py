"""
Встроенный 8-bit Adam (AdamW) без новых зависимостей.

Зачем: классический AdamW хранит 2 состояния (m, v) в fp32 = 8 байт на параметр.
Здесь оба состояния лежат в int8 с поблочным масштабом: ~2 байта на параметр,
т.е. экономия ~75% памяти оптимизатора. Работает везде (CPU/CUDA), в отличие
от bitsandbytes, которому нужны CUDA и Linux.

Как квантуем: блочное (blocksize=2048) companding-квантование —
x = sign * s^2 * absmax, где s линейно в int8. Квадратичная шкала даёт
высокую точность возле нуля (там живут почти все значения m и v) и грубую
возле максимума (там ошибка не страшна). Это упрощённый аналог dynamic
quantization из bitsandbytes, без таблиц и кастомных CUDA-ядер.

Если в системе есть bitsandbytes + CUDA — trainer сам выберет его
bnb.optim.AdamW8bit (там ещё точнее), иначе используется этот класс.
"""

import math

import torch
from torch.optim import Optimizer

BLOCKSIZE = 2048


def _quantize_blockwise(x: torch.Tensor, blocksize: int = BLOCKSIZE):
    """float32 -> (q int8, scales fp32). Возвращает плоские тензоры."""
    flat = x.detach().reshape(-1).float()
    n = flat.numel()
    pad = (-n) % blocksize
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.float32)])
    blocks = flat.reshape(-1, blocksize)
    absmax = blocks.abs().amax(dim=1).clamp_min_(1e-12)
    # companding: s = sign * sqrt(|x| / absmax) in [-1, 1]
    y = blocks.sign() * (blocks.abs() / absmax.unsqueeze(1)).sqrt()
    q = (y * 127.0).round().clamp_(-127, 127).to(torch.int8)
    return q.reshape(-1), absmax, n


def _dequantize_blockwise(q: torch.Tensor, scales: torch.Tensor,
                          n: int, blocksize: int = BLOCKSIZE, out_shape=None):
    """(q int8, scales) -> float32."""
    y = q.float().reshape(-1, blocksize) / 127.0
    x = y.sign() * y.square() * scales.unsqueeze(1)
    flat = x.reshape(-1)[:n]
    if out_shape is not None:
        flat = flat.reshape(out_shape)
    return flat


class Adam8bit(Optimizer):
    """AdamW с 8-битными состояниями. API как у torch.optim.AdamW."""

    def __init__(self, params, lr=2e-5, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, blocksize: int = BLOCKSIZE):
        if lr < 0.0:
            raise ValueError(f"lr={lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, blocksize=blocksize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            bs = group["blocksize"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Adam8bit не поддерживает разреженные градиенты")
                grad = p.grad.float()
                state = self.state[p]

                if not state:
                    state["step"] = 0
                    z = torch.zeros_like(p.data.float()).reshape(-1)
                    q, s, n = _quantize_blockwise(z, bs)
                    state["exp_avg_q"] = q
                    state["exp_avg_s"] = s
                    state["exp_avg_sq_q"] = q.clone()
                    state["exp_avg_sq_s"] = s.clone()
                    state["n"] = n

                state["step"] += 1
                step = state["step"]
                shape = p.data.shape
                n = state["n"]

                m = _dequantize_blockwise(state["exp_avg_q"], state["exp_avg_s"], n, bs)
                v = _dequantize_blockwise(state["exp_avg_sq_q"], state["exp_avg_sq_s"], n, bs)
                g = grad.reshape(-1)

                m.mul_(b1).add_(g, alpha=1.0 - b1)
                v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

                bc1 = 1.0 - b1 ** step
                bc2 = 1.0 - b2 ** step
                denom = (v / bc2).sqrt().add_(eps)
                update = (m / bc1) / denom

                # Переквантовываем состояния обратно в 8 бит
                qm, sm, _ = _quantize_blockwise(m, bs)
                qv, sv, _ = _quantize_blockwise(v, bs)
                state["exp_avg_q"].copy_(qm)
                state["exp_avg_s"].copy_(sm)
                state["exp_avg_sq_q"].copy_(qv)
                state["exp_avg_sq_s"].copy_(sv)

                # Decoupled weight decay + шаг (в fp32, как обычный AdamW)
                pd = p.data.float().reshape(-1)
                if wd != 0:
                    pd.add_(pd, alpha=-lr * wd)
                pd.add_(update.reshape(-1), alpha=-lr)
                p.data.copy_(pd.reshape(shape))

        return loss


    # Базовый load_state_dict кастит ВСЕ тензоры состояний к dtype параметра —
    # наши int8-кванты после рестарта тихо стали бы fp32. Поэтому чиним типы явно.
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        with torch.no_grad():
            for p, s in self.state.items():
                if "exp_avg_q" in s:
                    s["exp_avg_q"] = s["exp_avg_q"].to(device=p.device, dtype=torch.int8)
                    s["exp_avg_sq_q"] = s["exp_avg_sq_q"].to(device=p.device, dtype=torch.int8)
                    s["exp_avg_s"] = s["exp_avg_s"].to(device=p.device, dtype=torch.float32)
                    s["exp_avg_sq_s"] = s["exp_avg_sq_s"].to(device=p.device, dtype=torch.float32)


if __name__ == "__main__":
    # Мини-проверка: сходимость на игрушечной регрессии
    torch.manual_seed(0)
    X = torch.randn(64, 8)
    w_true = torch.randn(8, 1)
    y = X @ w_true

    for cls, kw, name in ((torch.optim.AdamW, {}, "AdamW-fp32"),
                          (Adam8bit, {}, "Adam8bit")):
        w = torch.zeros(8, 1, requires_grad=True)
        opt = cls([w], lr=1e-2, **kw)
        for _ in range(300):
            opt.zero_grad()
            ((X @ w - y) ** 2).mean().backward()
            opt.step()
        print(f"{name}: final loss={((X @ w - y) ** 2).mean().item():.6f}")
