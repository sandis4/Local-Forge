"""
ORPO: Odds Ratio Preference Optimization (Hong et al., 2024).

Одностадийное обучение: обычный SFT-лосс на выбранном ответе (chosen)
плюс штраф за предпочтение отклонённого (rejected) — через отношение шансов.
Без reward-модели, без генерации текста во время обучения (в отличие от PPO),
без замороженной reference-копии модели (в отличие от DPO — экономия
целая копия весов в памяти). Цена: 2 форварда на шаг (chosen + rejected).

  L = NLL(chosen) + beta * ( -log σ( log( odds(chosen) / odds(rejected) ) ) )
  odds(y|x) = P(y|x) / (1 - P(y|x)),  P = exp(средний логprob ответа)

beta (в статье lambda) по умолчанию 0.1.
"""

import torch
import torch.nn.functional as F


def _log_odds(avg_logp: torch.Tensor) -> torch.Tensor:
    """log(P/(1-P)) по среднему логprob. avg_logp < 0 всегда (P<1)."""
    p = torch.exp(avg_logp).clamp(max=1.0 - 1e-7)
    return avg_logp - torch.log1p(-p)


def orpo_loss(avg_logp_chosen: torch.Tensor, avg_logp_rejected: torch.Tensor,
              nll_chosen: torch.Tensor, beta: float = 0.1):
    """Возвращает (total, sft_part, or_part) — всё скаляры."""
    log_odds_ratio = _log_odds(avg_logp_chosen) - _log_odds(avg_logp_rejected)
    or_part = -F.logsigmoid(beta * log_odds_ratio).mean()
    total = nll_chosen + beta * or_part
    return total, nll_chosen.detach(), or_part.detach()


def avg_logp_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Средний логprob ответных токенов (labels != -100).
    logits: (B, T, V) float, labels: (B, T). Возвращает (B,) — по одному на сэмпл."""
    logps = torch.log_softmax(logits.float(), dim=-1)  # в fp32 для точности
    true_ids = labels.clamp_min(0).unsqueeze(-1)
    token_logp = logps.gather(-1, true_ids).squeeze(-1)
    mask = (labels != -100).float()
    denom = mask.sum(dim=1).clamp_min(1)
    return (token_logp * mask).sum(dim=1) / denom


if __name__ == "__main__":
    # Проверка: loss конечен, градиент тянет chosen вверх / rejected вниз
    torch.manual_seed(0)
    B, T, V = 2, 16, 64
    lc = torch.randn(B, T, V, requires_grad=True)
    lr = torch.randn(B, T, V, requires_grad=True)
    lab = torch.randint(0, V, (B, T))
    nll = F.cross_entropy(lc.reshape(-1, V), lab.reshape(-1))
    total, sft, orp = orpo_loss(avg_logp_from_logits(lc, lab),
                                avg_logp_from_logits(lr, lab), nll, beta=0.1)
    total.backward()
    print(f"total={total.item():.4f} sft={sft.item():.4f} or={orp.item():.4f}")
    # направление: средний градиент по логитам chosen должен растить prob правильных токенов
    gc = lc.grad.gather(-1, lab.unsqueeze(-1)).squeeze(-1).mean().item()
    gr = lr.grad.gather(-1, lab.unsqueeze(-1)).squeeze(-1).mean().item()
    print(f"grad@true-token chosen={gc:.6f} (хотим <0, т.е. рост prob), rejected={gr:.6f} (хотим >0)")
    assert gc < 0 and gr > 0
    print("ORPO DIRECTION OK")
