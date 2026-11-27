"""
Main CLI launcher with full configuration options and live mode switcher.
"""

import argparse
import sys
import os

from config import TrainingConfig, SPEED_PRESETS, DEFAULT_MODEL_PATH
from trainer import CPUTrainer, banner

def main():
    parser = argparse.ArgumentParser(
        description="CPU LLM Fine-Tuner with Dynamic Throttle & Clean Chat Formatting",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Core settings
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to local model folder or HF repo ID")
    parser.add_argument("--train_file", type=str, default="database/train.jsonl", help="Train dataset JSONL path")
    parser.add_argument("--val_file", type=str, default="database/validation.jsonl", help="Validation dataset JSONL path")
    parser.add_argument("--output_dir", type=str, default="output_model", help="Where to save LoRA adapters")
    parser.add_argument("--resume", type=str, default=None, help="Resume LoRA weights from checkpoint folder (default=auto: <output_dir>/interactive_checkpoint)")

    # Throttle presets
    parser.add_argument("--preset", type=str, choices=list(SPEED_PRESETS.keys()), default="work",
                        help="Initial speed preset: 'game' (background/low CPU), 'work' (medium), 'max' (afk/full speed), 'turbo' (all physical cores)")

    # Advanced Training Parameters
    parser.add_argument("--threads", type=int, default=None, help="Explicit number of CPU threads (overrides preset)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for AdamW (2e-5 = safe start, 2e-4 = aggressive)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping limit")
    parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup share of total steps (0.05-0.1 for big datasets; 0 = use fixed --warmup_steps)")
    parser.add_argument("--identity_file", type=str, default="database/identity.jsonl", help="Identity dataset to mix in (e.g. database/identity_qwen.jsonl for Qwen-based Forge)")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence token length")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of dataset rows (for testing/quick run)")
    parser.add_argument("--batch_size", type=int, default=None, help="Per-device train batch size (None=preset/auto)")
    parser.add_argument("--grad_accum", type=int, default=None, help="Gradient accumulation steps (None=preset/auto)")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling factor")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps")

    # Anti-forgetting
    parser.add_argument("--replay_file", type=str, default=None, help="General dataset for rehearsal mixing (anti-forgetting), e.g. database/ru.jsonl")
    parser.add_argument("--replay_ratio", type=float, default=0.2, help="Share of replay samples mixed into training stream (0..1)")
    parser.add_argument("--kl_anchor", type=float, default=0.0, help="KL anchor weight to base model (0=off, 0.5-1.0=strict, ~2x slower)")

    # Multi-model / big-model loading
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="Weight dtype for base model")
    parser.add_argument("--no_grad_checkpointing", action="store_true", help="Disable gradient checkpointing (faster, more RAM)")
    parser.add_argument("--mmap", choices=["auto", "on", "off"], default="auto", help="mmap loading for models larger than RAM (auto=decide by size)")

    # Universal "any model on any hardware" mode (default ON)
    parser.add_argument("--auto", dest="auto", action="store_true", default=True, help="Auto-plan dtype/context/LoRA/LISA/optimizer for available RAM/VRAM/disk")
    parser.add_argument("--no_auto", dest="auto", action="store_false", help="Disable auto-planning, use flags exactly as given")
    parser.add_argument("--lisa_layers", type=int, default=None, help="LISA: train only N transformer layers at a time, rotate (0=off, None=auto)")
    parser.add_argument("--lisa_interval", type=int, default=None, help="LISA rotation period in steps (None=auto)")
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "adamw8bit", "adafactor", "sgd", "galore"], help="Optimizer: adamw=exact, adamw8bit=8-bit states (~75%% RAM saving), adafactor/sgd=lightest, galore=full-weights low-rank states (None=auto)")
    parser.add_argument("--offload_dir", type=str, default=None, help="Disk offload folder for huge models (None=auto './offload')")
    parser.add_argument("--amp", dest="amp", action="store_true", default=True, help="Mixed precision forward: fp16 on CUDA, bf16 on CPU (default ON)")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable mixed precision (pure fp32)")
    parser.add_argument("--amp_dtype", type=str, default="auto", choices=["auto", "bf16", "fp16"], help="AMP dtype (auto=fp16 CUDA / bf16 CPU)")
    parser.add_argument("--tune_mode", type=str, default=None, choices=["lora", "full"], help="lora=adapters (default), full=train 100%% base weights directly (for GaLore)")
    parser.add_argument("--galore_rank", type=int, default=None, help="GaLore gradient projection rank (None=auto 128)")
    parser.add_argument("--galore_gap", type=int, default=None, help="GaLore SVD projector refresh every N steps (None=auto 200)")
    parser.add_argument("--galore_scale", type=float, default=None, help="GaLore low-rank step scale (None=auto 1.0)")
    parser.add_argument("--objective", type=str, default=None, choices=["sft", "orpo", "heretic"], help="sft=instruction tuning, orpo=SFT+preferences in one stage (needs --orpo_file), heretic=ARA censorship removal (no gradient training)")
    parser.add_argument("--orpo_file", type=str, default=None, help="JSONL with chosen/rejected pairs for ORPO")
    parser.add_argument("--orpo_beta", type=float, default=None, help="ORPO odds-ratio weight (None=auto 0.1)")
    parser.add_argument("--heretic_trials", type=int, default=None, help="Heretic: number of ARA optimization trials (default 3)")
    parser.add_argument("--heretic_startup_trials", type=int, default=None, help="Heretic: random exploration trials (default 1)")
    parser.add_argument("--heretic_quant", type=str, default=None, choices=["auto", "none", "bnb4"], help="Heretic: auto=bnb4 on CPU, none on CUDA")
    parser.add_argument("--resume_state", type=str, default=None, help="Continue from exact step: path to state file/dir (default=auto <output_dir>/training_state)")
    parser.add_argument("--no_resume_state", dest="resume_state", action="store_const", const="off", help="Start from scratch: do NOT load saved state (it will still be saved for the next night)")
    parser.add_argument("--no_state_save", dest="state_save", action="store_false", default=None, help="Do not write training state at all (breaks exact continuation)")
    parser.add_argument("--resume_step", type=str, default=None, help="Emergency: continue from step N (or 'auto' = latest checkpoint-step-N) without a state file; weights must be loaded via --resume")
    parser.add_argument("--cache_dir", type=str, default=None, help="Tokenized dataset cache dir (default=database/.tokcache)")
    parser.add_argument("--no_cache", dest="cache_dir", action="store_const", const="off", help="Disable tokenization cache")
    parser.add_argument("--compile", dest="compile", action="store_true", default=None, help="torch.compile the model (needs C++ compiler; pays off on Linux/GPU, not with frequent LISA)")
    parser.add_argument("--lora_plus_ratio", type=float, default=None, help="LoRA+: B-matrix LR multiplier (16 = paper's pick, 0/off; ~2x convergence, same step cost)")
    parser.add_argument("--attn", type=str, default=None, choices=["auto", "sdpa", "fa2"], help="Attention backend: auto=sdpa CPU / fa2 on CUDA if flash-attn installed (None=auto)")
    parser.add_argument("--liger", dest="liger", action="store_true", default=None, help="Patch model with Liger Triton kernels (+20%% speed, -60%% mem; Linux+CUDA only)")
    parser.add_argument("--quant4", type=str, default=None, choices=["auto", "on", "off"], help="4-bit NF4 base weights: auto=on CUDA when bf16 exceeds VRAM (None=auto)")
    parser.add_argument("--packing", action="store_true", help="Склеивать диалоги в одно окно max_length (между документами блочная маска внимания)")
    parser.add_argument("--group_by_length", action="store_true", help="Сортировать сэмплы по длине внутри окна — убирает разброс длин в батче")
    parser.add_argument("--length_window", type=int, default=64, help="Размер окна для length grouping (сколько сэмплов сортируются вместе)")
    parser.add_argument("--kd", action="store_true", help="Дистилляция: KL-лосс по top-k шансам учителя (поле soft в данных)")
    parser.add_argument("--kd_weight", type=float, default=0.5, help="Вес KD-лосса относительно обычного CE (0..1.5)")
    parser.add_argument("--kd_top_k", type=int, default=8, help="Сколько top-k токенов учителя держать в soft-targets")

    args = parser.parse_args()

    # Какие флаги пользователь задал явно — они всегда побеждают автоплан
    explicit = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}

    config = TrainingConfig(
        model_name_or_path=args.model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        identity_file=args.identity_file,
        max_length=args.max_length,
        max_samples=args.max_samples,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        save_steps=args.save_steps,
        replay_file=args.replay_file,
        replay_ratio=args.replay_ratio,
        kl_anchor=args.kl_anchor,
        dtype=args.dtype,
        gradient_checkpointing=not args.no_grad_checkpointing,
        force_mmap={"auto": None, "on": True, "off": False}[args.mmap]
    )

    # Apply speed preset
    config.apply_preset(args.preset)

    # Универсальный автоплан: любая модель на любом железе.
    # Применяем план только к тем полям, которые пользователь НЕ задал явно.
    if args.auto:
        from hardware import detect_hardware, describe_hardware
        from memory_plan import plan_for_model, describe_plan
        hw = detect_hardware()
        banner()
        print(" ЖЕЛЕЗО:")
        print(describe_hardware(hw))
        plan = plan_for_model(args.model, hw, user={})
        print(" АВТОПЛАН:")
        print(describe_plan(plan))
        for w in plan.get("warnings", []):
            print(f"  [!] {w}")
        banner()
        auto_fields = {
            "--max_length": ("max_length", plan["max_length"]),
            "--lora_r": ("lora_r", plan["lora_r"]),
            "--lora_alpha": ("lora_alpha", plan["lora_alpha"]),
            "--grad_accum": ("gradient_accumulation_steps", plan["gradient_accumulation_steps"]),
            "--batch_size": ("batch_size", plan["batch_size"]),
            "--kl_anchor": ("kl_anchor", plan["kl_anchor"]),
            "--mmap": ("force_mmap", plan["force_mmap"]),
            "--optimizer": ("optimizer", plan["optimizer"]),
            "--lisa_layers": ("lisa_layers", plan["lisa_layers"]),
            "--lisa_interval": ("lisa_interval", plan["lisa_interval"]),
            "--offload_dir": ("offload_dir", plan["offload_dir"]),
            "--amp_dtype": ("amp_dtype", plan["amp_dtype"]),
            "--save_steps": ("save_steps", plan["save_steps"]),
            "--quant4": ("quant4", plan["quant4"]),
        }
        for flag, (attr, val) in auto_fields.items():
            if flag not in explicit and val is not None:
                setattr(config, attr, val)
        if "--no_amp" not in explicit and "--amp" not in explicit:
            config.amp = plan["amp"]
        if "--no_grad_checkpointing" not in explicit:
            config.gradient_checkpointing = plan["gradient_checkpointing"]
        if plan.get("force_mmap") and "--mmap" not in explicit:
            config.force_mmap = True
        try:
            os.makedirs(config.offload_dir, exist_ok=True)
        except Exception:
            pass
        config.auto = True
    else:
        config.auto = False

    # Явные переопределения поверх автоплана: CLI всегда важнее плана.
    simple_overrides = [
        (args.lisa_layers, "lisa_layers"),
        (args.lisa_interval, "lisa_interval"),
        (args.optimizer, "optimizer"),
        (args.offload_dir, "offload_dir"),
        (args.tune_mode, "tune_mode"),
        (args.galore_rank, "galore_rank"),
        (args.galore_gap, "galore_gap"),
        (args.galore_scale, "galore_scale"),
        (args.objective, "objective"),
        (args.orpo_file, "orpo_file"),
        (args.orpo_beta, "orpo_beta"),
        (args.resume_state, "resume_state"),
        (args.resume_step, "resume_step"),
        (args.lora_plus_ratio, "lora_plus_ratio"),
        (args.attn, "attn"),
        (args.length_window, "length_window"),
    ]
    for value, attr in simple_overrides:
        if value is not None:
            setattr(config, attr, value)
    if "--no_amp" in explicit:
        config.amp = False
    elif "--amp" in explicit:
        config.amp = True
    if "--amp_dtype" in explicit:
        config.amp_dtype = args.amp_dtype
    if args.heretic_trials is not None:
        config.heretic_trials = max(1, args.heretic_trials)
    if args.heretic_startup_trials is not None:
        config.heretic_startup_trials = max(0, args.heretic_startup_trials)
    if args.heretic_quant is not None:
        config.heretic_quant = args.heretic_quant
    if args.state_save is not None:
        config.state_save = bool(args.state_save)
    if args.cache_dir is not None:
        config.cache_dir = None if args.cache_dir == "off" else args.cache_dir
    if args.compile is not None:
        config.compile = bool(args.compile)
    if args.liger is not None:
        config.liger = bool(args.liger)
    if args.quant4 is not None:
        config.quant4 = {"on": True, "off": False, "auto": "auto"}[args.quant4]
    if args.packing:
        config.packing = True
    if args.group_by_length:
        config.group_by_length = True
    if args.kd:
        config.kd = True
        config.kd_weight = max(0.0, min(2.0, args.kd_weight))
        config.kd_top_k = max(1, min(32, args.kd_top_k))

    if args.threads is not None:
        config.cpu_threads = args.threads
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.grad_accum is not None:
        config.gradient_accumulation_steps = args.grad_accum

    # Heretic — отдельный пайплайн: модель меняется через ARA, Trainer не запускается.
    if config.objective == "heretic":
        from heretic_runner import run_heretic
        run_heretic(config)
        return

    # Согласование новых режимов
    if config.objective == "orpo" and not config.orpo_file:
        raise SystemExit("[ORPO] Укажи --orpo_file с парами chosen/rejected "
                         "(иначе используй --objective sft).")
    if config.tune_mode == "full":
        if "--lisa_layers" not in explicit and config.lisa_layers == 0:
            config.lisa_layers = 2
            print("[Авто] Full-режим без LISA уронит память на всех слоях сразу — ставлю --lisa_layers 2.")
        if "--optimizer" not in explicit:
            config.optimizer = "galore"
            print("[Авто] Full-режим: оптимизатор по умолчанию — galore.")
        if config.optimizer in ("adamw", "adamw8bit"):
            print(f"[Подсказка] Full-режим с {config.optimizer} хранит полные состояния "
                  f"всех слоёв — рассмотри --optimizer galore.")
    if config.optimizer == "galore" and config.tune_mode == "lora" and "--tune_mode" not in explicit:
        print("[Подсказка] GaLore в lora-режиме работает, но раскрывается в --tune_mode full "
              "(там проекции считаются для весов базы, а не маленьких адаптеров).")

    # Resume весов: явный --resume важнее всего, иначе <output_dir>/interactive_checkpoint.
    # Старый дефолт всегда смотрел в output_model — из-за этого раны с кастомным
    # --output_dir (например output_forge1) молча стартовали с нуля.
    if args.resume is not None:
        if os.path.exists(args.resume):
            resume_path = args.resume
        else:
            resume_path = None
            print(f"[Resume] Папка '{args.resume}' не найдена — старт с нуля.")
    else:
        auto_resume = os.path.join(config.output_dir, "interactive_checkpoint")
        resume_path = auto_resume if os.path.exists(auto_resume) else None

    banner()
    print(" ЗАПУСК ОБУЧЕНИЯ НА CPU (LITTLE FORGE 1)")
    print(f" Модель: {config.model_name_or_path}")
    print(f" Профиль скорости: {args.preset} -> {SPEED_PRESETS[args.preset]['name']}")
    print(f" Количество ядер CPU: {config.cpu_threads}")
    print(f" Приоритет процесса в Windows: {config.process_priority}")
    print(f" Оптимизатор: {config.optimizer} | LISA слоёв: {config.lisa_layers} "
          f"(ротация {config.lisa_interval}) | Контекст: {config.max_length} | "
          f"LoRA r={config.lora_r} alpha={config.lora_alpha} | accum={config.gradient_accumulation_steps}")
    print(f" AMP: {'вкл (' + config.amp_dtype + ')' if config.amp else 'выкл'} | "
          f"Чекпоинтинг: {'вкл' if config.gradient_checkpointing else 'выкл'}")
    if config.packing:
        print(f" Padding: PACKING — диалоги склеены в окно {config.max_length} "
              f"(блочная маска между документами)")
    elif config.group_by_length:
        print(f" Padding: LENGTH GROUPING — сортировка по длине, окно {config.length_window}")
    else:
        print(" Padding: динамический (без packing и length grouping)")
    if config.kd:
        print(f" Дистилляция: KD weight={config.kd_weight:g}, top-k={config.kd_top_k} "
              f"(нужен датасет с полем soft)")
    print(f" Режим весов: {config.tune_mode} | Цель: {config.objective}"
          + (f" (beta={config.orpo_beta}, файл {config.orpo_file})" if config.objective == "orpo" else "")
          + (f" | GaLore r={config.galore_rank} gap={config.galore_gap}" if config.optimizer == "galore" else ""))
    print(f" Продолжение: {'выкл (--no_resume_state)' if config.resume_state == 'off' else config.resume_state}")
    if resume_path:
        print(f" Режим: Продолжение дообучения из '{resume_path}'")
    else:
        print(" Режим: Обучение с чистого листа")
    banner()

    trainer = CPUTrainer(config, resume_checkpoint=resume_path)
    trainer.train()

if __name__ == "__main__":
    main()
