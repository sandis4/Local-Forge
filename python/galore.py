"""
GaLore: Gradient Low-Rank Projection (Zhao et al., 2024).

Идея: обучаем 100% весов (качество выше, чем у LoRA), но состояния
оптимизатора (m, v) храним не для полной матрицы градиентов m×n,
а для её низкоранговой проекции r×k. Проектор (топ-r сингулярные векторы
самого градиента) обновляется через SVD каждые update_gap шагов.

Память на большой матрице: 2·m·n (fp32 Adam) -> ~2·r·max(m,n) + m·r.
При m=n=4096, r=128: ~3% от исходных состояний. В статье: обучение 7B
в 24 ГБ вместо 60+ ГБ (всего −65% памяти).

Здесь: чистый torch, без зависимостей, работает на CPU и CUDA.
Параметры 1D и матрицы меньше ранга идут обычным AdamW (фолбэк).
Отлично сочетается с LISA тренера: проекции считаются только
для активных слоёв.
"""

import torch
from torch.optim import Optimizer


class GaLoreAdamW(Optimizer):
    """AdamW с GaLore-проекцией градиентов. API как у torch.optim.AdamW."""

    def __init__(self, params, lr=2e-5, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, rank: int = 128, update_gap: int = 200,
                 scale: float = 1.0):
        if lr < 0.0:
            raise ValueError(f"lr={lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        rank=rank, update_gap=update_gap, scale=scale)
        super().__init__(params, defaults)

    @staticmethod
    def _projector(grad_2d: torch.Tensor, rank: int, side: str):
        """Топ-rank сингулярные векторы градиента. side='left' -> P (m×r),
        side='right' -> Q (n×r). Намеренно детерминированный полный SVD
        (без svd_lowrank): иначе его внутренний RNG ломал бы точное
        продолжение обучения с шага — проектор после resume отличался бы."""
        U, _, Vh = torch.linalg.svd(grad_2d.float(), full_matrices=False)
        if side == "left":
            return U[:, :rank]
        return Vh[:, :rank]

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
            rank = group["rank"]
            gap = group["update_gap"]
            scale = group["scale"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("GaLore не поддерживает разреженные градиенты")
                grad = p.grad.float()
                state = self.state[p]
                if not state:
                    state["step"] = 0

                state["step"] += 1
                step = state["step"]
                shape = p.data.shape

                # Малые/плоские параметры — обычный AdamW без проекции
                if grad.dim() != 2 or min(shape) <= rank:
                    m = state.get("m", torch.zeros_like(grad))
                    v = state.get("v", torch.zeros_like(grad))
                    m.mul_(b1).add_(grad, alpha=1.0 - b1)
                    v.mul_(b2).addcmul_(grad, grad, value=1.0 - b2)
                    state["m"], state["v"] = m, v
                    upd = (m / (1.0 - b1 ** step)) / ((v / (1.0 - b2 ** step)).sqrt() + eps)
                    pd = p.data.float()
                    if wd != 0:
                        pd.add_(pd, alpha=-lr * wd)
                    pd.add_(upd, alpha=-lr)
                    p.data.copy_(pd.reshape(shape))
                    continue

                m, n = shape
                side = "left" if m < n else "right"

                # Обновляем проектор каждые gap шагов (или при смене формы)
                key = "P" if side == "left" else "Q"
                if key not in state or (step % gap == 1) or state[key].shape[0] != (m if side == "left" else n):
                    state[key] = self._projector(grad.reshape(m, n), rank, side)
                    state["side"] = side
                proj = state[key]

                G = grad.reshape(m, n)
                R = proj.t() @ G if side == "left" else G @ proj  # (r×n) или (m×r)

                Rm = state.get("Rm", torch.zeros_like(R))
                Rv = state.get("Rv", torch.zeros_like(R))
                Rm.mul_(b1).add_(R, alpha=1.0 - b1)
                Rv.mul_(b2).addcmul_(R, R, value=1.0 - b2)
                state["Rm"], state["Rv"] = Rm, Rv

                N = (Rm / (1.0 - b1 ** step)) / ((Rv / (1.0 - b2 ** step)).sqrt() + eps)
                back = (proj @ N) if side == "left" else (N @ proj.t())  # снова m×n

                pd = p.data.float()
                if wd != 0:
                    pd.add_(pd.reshape(m, n), alpha=-lr * wd)
                pd.reshape(m, n).add_(back, alpha=-lr * scale)
                p.data.copy_(pd.reshape(shape))

        return loss

    # Базовый load_state_dict кастит все тензоры к dtype параметра —
    # проекторы и моменты возвращаем в fp32 явно, иначе resume сломает математику.
    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        with torch.no_grad():
            for p, s in self.state.items():
                for k in ("P", "Q", "Rm", "Rv", "m", "v"):
                    if k in s and isinstance(s[k], torch.Tensor):
                        s[k] = s[k].to(device=p.device, dtype=torch.float32)


if __name__ == "__main__":
    # Мини-проверка: полноранговая регрессия 64×64, ранг проекции 8
    torch.manual_seed(0)
    X = torch.randn(128, 64)
    Wt = torch.randn(64, 64)
    Y = X @ Wt
    for cls, kw, name in ((torch.optim.AdamW, {}, "AdamW"),
                          (GaLoreAdamW, {"rank": 8, "update_gap": 20}, "GaLore r=8")):
        W = torch.zeros(64, 64, requires_grad=True)
        opt = cls([W], lr=3e-3, **kw)
        for _ in range(400):
            opt.zero_grad()
            ((X @ W - Y) ** 2).mean().backward()
            opt.step()
        print(f"{name}: final loss={((X @ W - Y) ** 2).mean().item():.6f}")
