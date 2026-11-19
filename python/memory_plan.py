"""
Планировщик обучения под любое железо и любой размер модели.

Идея: по размеру весов (ГБ из safetensors-индекса) и доступной RAM/VRAM
автоматически подбираем стратегию, чтобы обучение вообще запустилось:
  - ram: модель влезает в RAM — обычное QLoRA/LoRA обучение;
  - mmap: веса больше ~50% свободной RAM — zero-copy загрузка с диска
    (уже есть в model_loader.py), плюс послойное обучение LISA,
    экономный оптимизатор и укороченный контекст;
  - disk: веса больше всей свободной RAM — то же самое, но ещё строже
    (1-2 активных слоя, короткий контекст, большое накопление градиентов).

LoRA-адаптеры сами по себе крошечные (десятки-сотни МБ даже для 120B),
поэтому "любая модель на любом железе" достигается комбинацией:
mmap/disc-offload базы + LISA-ротация активных слоёв + gradient checkpointing
+ Adafactor/SGD вместо AdamW + микро-батч 1. Платим скоростью (на CPU
шаг 120B-модели — минуты), но обучение идёт, а не падает с OOM.

Никаких новых зависимостей: только psutil/torch/transformers, которые уже есть.
"""

import glob
import os

from hardware import detect_hardware

_DTYPE_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
                "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def estimate_weights_gb(model_dir):
    """Быстрая оценка размера весов по *.safetensors без загрузки модели.
    Возвращает (gb, n_files). Для .bin/.gguf — грубая оценка по размеру файлов."""
    paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if paths:
        total = 0
        try:
            from safetensors import safe_open
            for p in paths:
                try:
                    with safe_open(p, framework="pt", device="cpu") as f:
                        for k in f.keys():
                            sl = f.get_slice(k)
                            shape, dt = sl.get_shape(), str(sl.get_dtype())
                            numel = 1
                            for s in shape:
                                numel *= s
                            total += numel * _DTYPE_SIZES.get(dt, 4)
                except Exception:
                    # битый/занятый шард — оценим по размеру файла
                    try:
                        total += os.path.getsize(p)
                    except OSError:
                        pass
            return total / 1e9, len(paths)
        except Exception:
            total = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
            return total / 1e9, len(paths)
    # fallback: любые файлы весов
    cand = []
    for ext in ("*.bin", "*.gguf", "*.pt", "*.pth"):
        cand += glob.glob(os.path.join(model_dir, ext))
    if cand:
        total = sum(os.path.getsize(p) for p in cand if os.path.exists(p))
        return total / 1e9, len(cand)
    return 0.0, 0


def plan_for_model(model_dir, hw=None, user=None):
    """Строит план обучения. user — dict явных переопределений из CLI/config.

    Возвращает dict с полями:
      strategy: 'ram' | 'mmap' | 'disk'
      weights_gb, dtype, force_mmap, gradient_checkpointing,
      max_length, lora_r, lora_alpha, lora_dropout,
      lisa_layers (0 = обучать все слои сразу), lisa_interval,
      optimizer ('adamw'|'adafactor'|'sgd'),
      batch_size, gradient_accumulation_steps, kl_anchor (0 = выкл),
      offload_dir, warnings (list[str])
    """
    hw = hw or detect_hardware()
    user = user or {}
    warnings = []

    weights_gb, n_files = estimate_weights_gb(model_dir)
    ram_avail = float(hw.get("ram_avail_gb") or 0)
    ram_total = float(hw.get("ram_total_gb") or 0)
    vram_free = float(hw.get("vram_free_gb") or 0)
    cuda = bool(hw.get("cuda_available"))

    # Эффективный бюджет: VRAM + половина свободной RAM (вторую половину
    # оставляем системе/браузеру/играм). На CPU-машине — только RAM.
    budget = (vram_free + 0.5 * ram_avail) if cuda else (0.5 * ram_avail if ram_avail else 0)
    ratio = (weights_gb / budget) if budget > 0 else 99.0

    if weights_gb <= 0:
        warnings.append("Не найдены веса (*.safetensors) — план по умолчанию (RAM).")
        strategy = "ram"
    elif ratio <= 1.0 and weights_gb <= 0.5 * (ram_avail or 1e9):
        strategy = "ram"
    elif weights_gb <= (ram_avail or 0) * 1.1:
        strategy = "mmap"
    else:
        strategy = "disk"

    if strategy == "ram":
        plan = dict(
            gradient_checkpointing=True, max_length=512,
            lora_r=8, lora_alpha=16, lora_dropout=0.05,
            lisa_layers=0, lisa_interval=50,
            optimizer="adamw", batch_size=1,
            gradient_accumulation_steps=8, kl_anchor=0.0,
            save_steps=50,
            amp=True, amp_dtype="auto",
        )
    elif strategy == "mmap":
        warnings.append(
            f"Веса {weights_gb:.1f} ГБ велики для свободной RAM {ram_avail:.1f} ГБ: "
            "включаю mmap + LISA (2 слоя) + Adam8bit + AMP + контекст 256. Будет медленно, но без OOM."
        )
        plan = dict(
            gradient_checkpointing=True, max_length=256,
            lora_r=8, lora_alpha=16, lora_dropout=0.05,
            lisa_layers=2, lisa_interval=50,
            optimizer="adamw8bit", batch_size=1,
            gradient_accumulation_steps=16, kl_anchor=0.0,
            save_steps=25,
            amp=True, amp_dtype="auto",
        )
    else:
        warnings.append(
            f"Веса {weights_gb:.1f} ГБ больше свободной RAM {ram_avail:.1f} ГБ: "
            "режим disk-stream (1 слой LISA, Adam8bit, AMP, контекст 128, accum 32). "
            "Шаг может занимать минуты на CPU — это нормально."
        )
        plan = dict(
            gradient_checkpointing=True, max_length=128,
            lora_r=4, lora_alpha=8, lora_dropout=0.05,
            lisa_layers=1, lisa_interval=30,
            optimizer="adamw8bit", batch_size=1,
            gradient_accumulation_steps=32, kl_anchor=0.0,
            save_steps=10,
            amp=True, amp_dtype="auto",
        )

    # CUDA-машина с 4-битными либами: можно поднять контекст/ранг
    if cuda and (hw.get("has_bitsandbytes") or hw.get("has_quanto")):
        plan["quant_4bit"] = True
        if strategy != "disk":
            plan["max_length"] = max(plan["max_length"], 512)
    else:
        plan["quant_4bit"] = False
        if cuda and weights_gb > 0:
            warnings.append("Для 4-бит на CUDA поставь bitsandbytes или quanto — пока будет mmap bf16/fp32.")

    # QLoRA-решение: 4-битная база, если bf16 не влезает в VRAM (на CPU всегда None=auto).
    # None здесь = "не трогать выбор пользователя/дефолт".
    plan["quant4"] = (True if (cuda and vram_free > 0 and weights_gb > 0.60 * (hw.get("vram_total_gb") or vram_free))
                      else None)
    if plan["quant4"]:
        warnings.append(f"База {weights_gb:.1f} ГБ не влезает в VRAM — включаю NF4-4bit (QLoRA).")

    # Явные переопределения пользователя — всегда побеждают авто
    for k in ("max_length", "lora_r", "lora_alpha", "lora_dropout",
              "lisa_layers", "lisa_interval", "optimizer",
              "batch_size", "gradient_accumulation_steps", "kl_anchor",
              "gradient_checkpointing", "dtype", "offload_dir",
              "amp", "amp_dtype", "tune_mode", "galore_rank", "galore_gap",
              "galore_scale", "objective", "orpo_file", "orpo_beta",
              "save_steps", "quant4"):
        if k in user and user[k] is not None:
            # 'auto' = не переопределять
            if isinstance(user[k], str) and user[k] == "auto":
                continue
            plan[k] = user[k]

    # Дефолты, если пользователь ничего не задал
    plan.setdefault("dtype", "auto")
    plan.setdefault("offload_dir", os.path.join(os.getcwd(), "offload"))

    plan["strategy"] = strategy
    plan["weights_gb"] = weights_gb
    plan["force_mmap"] = True if strategy in ("mmap", "disk") else None
    plan["warnings"] = warnings
    plan["budget_gb"] = budget
    return plan


def describe_plan(plan):
    lisa = plan.get("lisa_layers") or 0
    lisa_s = "выкл (все слои сразу)" if not lisa else f"{lisa} слоя, ротация каждые {plan.get('lisa_interval')} шагов"
    amp = plan.get("amp")
    amp_s = f"вкл ({plan.get('amp_dtype')})" if amp else "выкл"
    return (
        f"  Стратегия: {plan.get('strategy')} (веса {plan.get('weights_gb', 0):.1f} ГБ, бюджет {plan.get('budget_gb', 0):.1f} ГБ)\n"
        f"  Контекст: {plan.get('max_length')}, LoRA r={plan.get('lora_r')} alpha={plan.get('lora_alpha')}\n"
        f"  LISA: {lisa_s} | Оптимизатор: {plan.get('optimizer')} | AMP: {amp_s} | accum: {plan.get('gradient_accumulation_steps')}\n"
        f"  Чекпоинт каждые {plan.get('save_steps')} шагов | "
        f"mmap: {plan.get('force_mmap')} | checkpointing: {plan.get('gradient_checkpointing')} | "
        f"KL-якорь: {plan.get('kl_anchor')}"
    )


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gemma_base")
    args = ap.parse_args()
    hw = detect_hardware()
    from hardware import describe_hardware
    print(describe_hardware(hw))
    plan = plan_for_model(args.model, hw)
    print(describe_plan(plan))
    for w in plan["warnings"]:
        print("  [!] " + w)
    print(json.dumps({k: v for k, v in plan.items() if k != "warnings"},
                     indent=2, ensure_ascii=False, default=str))
