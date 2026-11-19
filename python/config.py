"""
Training presets and configuration management for Qwen 2.5 3B / Little Forge 1.
"""

from dataclasses import dataclass, field
import multiprocessing
import os

TOTAL_CPUS = multiprocessing.cpu_count()


def _physical_cores(total: int) -> int:
    """Физические ядра: SMT почти не ускоряет матричные операции,
    но заметно добавляет тепла и сбрасывает частоты на ноутбуках."""
    try:
        import psutil
        n = psutil.cpu_count(logical=False)
        if n and 0 < n <= total:
            return int(n)
    except Exception:
        pass
    return total


PHYSICAL_CORES = _physical_cores(TOTAL_CPUS)

# Speed & Resource Presets
SPEED_PRESETS = {
    "game": {
        "name": "Game / Background Mode (Игровой / Фоновый)",
        "cpu_threads": max(1, PHYSICAL_CORES // 2),
        "process_priority": "idle",
        "step_delay_sec": 0.35,
        "gradient_accumulation_steps": 8,
        "batch_size": 1,
    },
    "work": {
        "name": "Work / Browser Mode (Рабочий / Браузер)",
        "cpu_threads": max(2, PHYSICAL_CORES),
        "process_priority": "below_normal",
        "step_delay_sec": 0.05,
        "gradient_accumulation_steps": 8,   # усреднение градиентов за 8 шагов (стабильность)
        "batch_size": 1,
    },
    "max": {
        "name": "Max Speed / Afk Mode (Максимальная скорость / Отошел)",
        "cpu_threads": max(2, PHYSICAL_CORES - 1),  # одно ядро остаётся системе
        "process_priority": "normal",
        "step_delay_sec": 0.0,
        "gradient_accumulation_steps": 8,
        "batch_size": 1,
    },
    "turbo": {
        "name": "Turbo (Все физические ядра)",
        "cpu_threads": PHYSICAL_CORES,
        "process_priority": "high",
        "step_delay_sec": 0.0,
        "gradient_accumulation_steps": 8,
        "batch_size": 1,
    }
}

DEFAULT_MODEL_PATH = "models/gemma_base" if os.path.exists("models/gemma_base/model.safetensors") else "models/qwen2.5_3b"

@dataclass
class TrainingConfig:
    # Model & Paths
    model_name_or_path: str = DEFAULT_MODEL_PATH
    train_file: str = "database/train.jsonl"
    val_file: str = "database/validation.jsonl"
    output_dir: str = "output_model"

    # Универсальная загрузка (любые модели и размеры)
    dtype: str = "auto"                  # auto | bf16 | fp32
    gradient_checkpointing: bool = True  # экономия RAM на активациях
    force_mmap: bool = None              # None = авто по размеру; True/False — принудительно
    
    # Preset
    preset: str = "work"
    
    # Hardware & Speed
    cpu_threads: int = field(default_factory=lambda: max(2, PHYSICAL_CORES))
    step_delay_sec: float = 0.05
    process_priority: str = "below_normal"
    
    # Hyperparameters
    learning_rate: float = 2e-5      # безопасный старт: ровный лосс без скачков (было 2e-4)
    max_grad_norm: float = 1.0       # лимит градиентов (клиппинг)
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: int = 1
    max_length: int = 512
    max_samples: int = None
    warmup_steps: int = 20        # используется, если warmup_ratio = 0
    warmup_ratio: float = 0.05    # разогрев 5% от всего обучения (0.05-0.1 для больших датасетов)
    save_steps: int = 50
    logging_steps: int = 1
    
    # LoRA / PEFT for Qwen 2.5
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: any = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    # Assistant-only loss
    train_only_on_assistant: bool = True
    identity_file: str = "database/identity.jsonl"  # файл личности для подмешивания

    # Anti-forgetting (защита от катастрофического забывания)
    replay_file: str = None        # общий датасет для репетиции (например database/ru.jsonl)
    replay_ratio: float = 0.2      # доля репетиционных сэмплов от основного потока
    kl_anchor: float = 0.0         # вес KL-якоря к базовой модели (0 = выкл; 0.5-1.0 = строгая защита, медленнее)

    # Универсальный режим "любая модель на любом железе" (см. memory_plan.py)
    auto: bool = True              # автоплан под доступные RAM/VRAM/диск
    optimizer: str = "adamw"       # adamw | adamw8bit | adafactor | sgd (8bit — состояния int8, ~75% экономии)
    lisa_layers: int = 0           # 0 = обучать все слои сразу; N = LISA-ротация по N слоёв
    lisa_interval: int = 50        # менять активные слои каждые N шагов
    offload_dir: str = "offload"   # папка дискового offload для огромных моделей

    # Смешанная точность: форвард в половинной точности, шаги — в полной
    amp: bool = True               # AMP вкл (ускорение + меньше памяти под активации)
    amp_dtype: str = "auto"        # auto | bf16 | fp16 (auto: fp16 на CUDA, bf16 на CPU)

    # Режим весов: lora (адаптеры поверх замороженной базы) или
    # full (обучение 100% весов базы напрямую — под GaLore)
    tune_mode: str = "lora"
    galore_rank: int = 128         # ранг проекции градиентов (меньше = меньше памяти)
    galore_gap: int = 200          # пересчёт SVD-проектора каждые N шагов
    galore_scale: float = 1.0      # масштаб низкорангового шага

    # Цель: sft (инструкции), orpo (sft + предпочтения одним этапом)
    # или heretic (удаление цензуры через ARA, без градиентного обучения)
    objective: str = "sft"
    orpo_file: str = None          # jsonl пар chosen/rejected для ORPO
    orpo_beta: float = 0.1         # вес odds-ratio штрафа (0.05-0.5)

    # Heretic: оптимизация ARA. trials=3 — минимум для CPU; quantization:
    # auto (CPU -> bnb4, CUDA -> none), none, bnb4.
    heretic_trials: int = 3
    heretic_startup_trials: int = 1
    heretic_quant: str = "auto"

    # Точное продолжение с шага остановки: "auto" (<output_dir>/training_state),
    # путь к файлу/папке состояния или "off" (НЕ загружать состояние;
    # запись продолжается — иначе ночь нельзя будет продолжить)
    resume_state: str = "auto"
    state_save: bool = True         # писать файл состояния (чекпоинты/пауза/финал)
    resume_step: int = None         # аварийно: продолжить с шага N без файла состояния

    # Кэш токенизации датасета: первый проход строит, все скипы/рераны читают.
    # None/"off" = выкл (токенизировать всё заново каждый раз).
    cache_dir: str = "database/.tokcache"

    # torch.compile (индуктор): ускорение форварда. На этом Windows-ПК нет
    # C++ компилятора — упадёт в понятное предупреждение; раскрывается на
    # Linux/GPU (облако). Несовместим с частой LISA-ротацией (рекомпиляция):
    # включай только при lisa_layers=0 или большом lisa_interval.
    compile: bool = False

    # LoRA+ (ICML 2024): B-матрицам LR в ratio раз выше. До ~2x сходимости
    # и +1-2% качества при той же цене шага. 0 = выкл (классический LoRA).
    lora_plus_ratio: float = 0.0

    # Внимание: auto | sdpa | fa2 (FlashAttention-2, только Linux+CUDA).
    # На CPU всегда sdpa; fa2 без пакета flash-attn молча откатится.
    attn: str = "auto"

    # Liger kernels (RMSNorm/SwiGLU/CE в одном Triton-ядре): +20% throughput,
    # -60% памяти. Только Linux+CUDA+Triton; иначе молча пропускается.
    liger: bool = False

    # 4-битная база (QLoRA NF4): auto | on | off. На CPU игнорируется;
    # на CUDA включается само, когда bf16 не влезает в VRAM.
    quant4: str = "auto"

    # Packing: несколько диалогов склеиваются в одно окно max_length.
    # Между документами — блочная маска внимания, лосс только по токенам
    # ассистента. Даёт больше токенов на форвард. False = как раньше.
    packing: bool = False

    # Дистилляция (KD): учительские top-k шансы из поля soft в датасете.
    # KL-член добавляется к обычному CE; работает только для SFT.
    kd: bool = False
    kd_weight: float = 0.5
    kd_top_k: int = 8

    # Length grouping: сэмплы сортируются по длине внутри окна length_window,
    # поэтому в батче нет разброса длин и паддинга почти не остаётся.
    # С packing несовместимо по смыслу (packing уже убирает паддинг).
    group_by_length: bool = False
    length_window: int = 64

    def apply_auto_plan(self, plan: dict):
        """Применяет план из memory_plan.plan_for_model к конфигу."""
        for k in ("max_length", "lora_r", "lora_alpha", "lora_dropout",
                  "lisa_layers", "lisa_interval", "optimizer",
                  "batch_size", "gradient_accumulation_steps", "kl_anchor",
                  "gradient_checkpointing", "dtype", "offload_dir",
                  "amp", "amp_dtype", "tune_mode", "galore_rank", "galore_gap",
                  "galore_scale", "objective", "orpo_file", "orpo_beta",
                  "save_steps", "quant4"):
            if k in plan and plan[k] is not None:
                setattr(self, k, plan[k])
        if plan.get("force_mmap") is not None:
            self.force_mmap = plan["force_mmap"]

    def apply_preset(self, preset_name: str):
        if preset_name in SPEED_PRESETS:
            p = SPEED_PRESETS[preset_name]
            self.preset = preset_name
            self.cpu_threads = p["cpu_threads"]
            self.process_priority = p["process_priority"]
            self.step_delay_sec = p["step_delay_sec"]
            self.gradient_accumulation_steps = p["gradient_accumulation_steps"]
            self.batch_size = p["batch_size"]
            return True
        return False
