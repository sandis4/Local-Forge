"""
CPU Trainer engine with streaming data loading and minimal RAM footprint.
Includes dynamic throttling, Forge Identity Template baking, and interactive console controls.
"""

import os
import sys
import time
import threading
import gc
import glob
import gzip
import json
import random
import re
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)

from config import TrainingConfig, SPEED_PRESETS
from resource_manager import DynamicResourceManager
from dataset_formatter import (StreamingConversationDataset, ReplayMixedDataset, collate_fn,
                              LengthGroupedDataset, PackedDataset, _open_text)
from model_loader import load_model_auto

# Проекции, которые затыкаем LoRA (точное совпадение последнего компонента имени модуля)
LORA_PROJ_NAMES = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
# Куски имён мультимодальных башен — туда LoRA не заходим
VISION_AUDIO_HINTS = ("visual", "vision", "audio", "multi_modal", "projector")

LITTLE_FORGE_DEFAULT_PROMPT = (
    "You are Forge, an intelligent and helpful AI assistant fine-tuned by "
    "Reforge Squad based on the Gemma 4 E2B architecture and the Fable-5-Premium dataset."
)


def kd_loss_from_logits(logits, soft_ids, soft_probs, soft_mask, labels=None, chunk=16):
    """KL(teacher || student) по top-k учителя, без полного softmax по словарю.

    logits[i] предсказывает токен i+1, поэтому сдвигаем soft-targets и labels
    на один шаг. Считаем чанками по позициям: на CPU это экономит гигабайты.
    """
    if logits is None or soft_ids is None or soft_mask is None:
        return None
    logits = logits[:, :-1, :]
    ids = soft_ids[:, 1:, :].to(logits.device)
    probs = soft_probs[:, 1:, :].to(device=logits.device, dtype=torch.float32)
    mask = soft_mask[:, 1:].to(device=logits.device, dtype=torch.float32)
    if labels is not None:
        mask = mask * (labels[:, 1:] != -100).float()

    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    count = torch.zeros((), dtype=torch.float32, device=logits.device)
    seq = logits.size(1)
    for i in range(0, seq, chunk):
        m = mask[:, i:i + chunk]
        if not m.any():
            continue
        lc = logits[:, i:i + chunk].float()
        sid = ids[:, i:i + chunk].clamp(min=0, max=lc.size(-1) - 1)
        q = probs[:, i:i + chunk]
        gathered = lc.gather(-1, sid)
        logp = gathered - torch.logsumexp(lc, dim=-1, keepdim=True)
        kl = -(q * logp).sum(-1)
        total = total + (kl * m).sum()
        count = count + m.sum()
    if count.item() <= 0:
        return None
    return total / count


def _file_has_soft(path):
    try:
        with _open_text(path) as f:
            for n, line in enumerate(f):
                if n >= 20:
                    break
                line = line.strip()
                if not line:
                    continue
                if "soft" in json.loads(line):
                    return True
    except Exception:
        pass
    return False

GEMMA4_LITTLE_FORGE_TEMPLATE = (
    "{{ bos_token }}"
    "{%- if messages[0]['role'] == 'system' -%}"
        "{{ '<|turn>system\\n' + messages[0]['content'] + '<turn|>\\n' }}"
        "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}"
        "{{ '<|turn>system\\nYou are Forge, an intelligent and helpful AI assistant created and fine-tuned by the Reforge Squad team on top of the Gemma 4 E2B architecture using the Fable-5-Premium dataset and SWE-smith software engineering data. Always answer in the user\\'s language.<turn|>\\n' }}"
        "{%- set loop_messages = messages -%}"
    "{%- endif -%}"
    "{%- for message in loop_messages -%}"
        "{%- if message['role'] == 'user' -%}"
            "{{ '<|turn>user\\n' + message['content'] + '<turn|>\\n' }}"
        "{%- elif message['role'] == 'assistant' or message['role'] == 'model' -%}"
            "{{ '<|turn>model\\n' + message['content'] + '<turn|>\\n' }}"
        "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
        "{{ '<|turn>model\\n' }}"
    "{%- endif -%}"
)

class CPUTrainer:
    def __init__(self, config: TrainingConfig, resume_checkpoint: str = None):
        self.config = config
        self.resume_checkpoint = resume_checkpoint
        self.res_mgr = DynamicResourceManager(
            cpu_threads=config.cpu_threads,
            priority=config.process_priority,
            step_delay=config.step_delay_sec
        )
        self.stop_requested = False
        self._tune_mode = getattr(config, "tune_mode", "lora")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._opt = None          # текущий оптимизатор (для команды save и state-снапшотов)
        self._sched = None        # текущий шедулер
        self._global_step = 0
        self._cur_epoch = 0
        self._cur_epoch_batches = 0
        self._cur_line = 0
        self._kd_last = None
        
        print("=" * 60)
        print(" ИНИЦИАЛИЗАЦИЯ ТРЕНЕРА ДЛЯ CPU (LITTLE FORGE 1 / REFORGE SQUAD)")
        print(f" Модель / Путь: {config.model_name_or_path}")
        print(f" Стартовый профиль: {config.preset}")
        print(f" Потоков CPU: {config.cpu_threads}")
        print(f" Приоритет процесса: {config.process_priority}")
        if self.resume_checkpoint and os.path.exists(self.resume_checkpoint):
            print(f" Продолжение дообучения с чекпоинта: {self.resume_checkpoint}")
        print("=" * 60)

    def _detect_model_family(self) -> str:
        """Определяем семейство модели по config.json — для условной логики (шаблоны и т.п.)."""
        try:
            import json as _json
            cfg_path = os.path.join(self.config.model_name_or_path, "config.json")
            with open(cfg_path, encoding="utf-8") as f:
                cfg = _json.load(f)
            mt = (cfg.get("model_type") or "") + " " + " ".join(cfg.get("architectures") or [])
            mt = mt.lower()
            if "gemma" in mt:
                return "gemma"
            if "qwen" in mt:
                return "qwen"
            if "llama" in mt:
                return "llama"
            return mt.split()[0] if mt.split() else "unknown"
        except Exception:
            return "unknown"

    # ---- LISA: послойное обучение (любая модель на любом железе) ----
    def _is_rotated_param(self, name: str) -> bool:
        """Параметр участвует в LISA-ротации? В lora-режиме — LoRA-матрицы,
        в full-режиме — веса проекций самой базы (для GaLore full-finetune)."""
        if self._tune_mode == "full":
            parts = name.split(".")
            if len(parts) < 2 or parts[-1] != "weight":
                return False
            if parts[-2] not in LORA_PROJ_NAMES:
                return False
            if any(h in name.lower() for h in VISION_AUDIO_HINTS):
                return False
            return True
        return "lora_" in name

    def _collect_lisa_groups(self):
        """Группирует обучаемые LoRA-параметры по индексу трансформерного слоя.
        Возвращает dict: layer_idx -> [param_names]. Слои без номера (embeddings,
        lm_head) складываем в группу -1 — она всегда активна."""
        groups = {}
        for name, p in self.model.named_parameters():
            if not p.requires_grad or not self._is_rotated_param(name):
                continue
            m = re.search(r"\.layers?\.(\d+)\.", name)
            idx = int(m.group(1)) if m else -1
            groups.setdefault(idx, []).append(name)
        self._lisa_groups = groups
        return groups

    def _apply_lisa(self, active):
        """active: set с индексами слоёв. Группа -1 (embeddings/head) всегда активна.
        Замороженным LoRA-параметрам ставим requires_grad=False — они не тратят
        память на градиенты и не участвуют в backward."""
        active = set(active) | {-1}
        frozen, unfrozen = 0, 0
        for name, p in self.model.named_parameters():
            if not self._is_rotated_param(name):
                continue
            m = re.search(r"\.layers?\.(\d+)\.", name)
            idx = int(m.group(1)) if m else -1
            if idx in active:
                p.requires_grad = True
                unfrozen += 1
            else:
                p.requires_grad = False
                frozen += 1
        self._lisa_active = sorted(active)
        return frozen, unfrozen

    def _rotate_lisa(self, global_step):
        """Случайная ротация активных слоёв (как в LISA-препринте)."""
        n = self.config.lisa_layers
        layers = sorted(k for k in self._lisa_groups if k >= 0)
        if not layers or n <= 0:
            return
        n = min(n, len(layers))
        active = set(random.sample(layers, n))
        frozen, unfrozen = self._apply_lisa(active)
        print(f"[LISA шаг {global_step}] активные слои: {sorted(active)} "
              f"(заморожено обучаемых параметров: {frozen}, активно: {unfrozen})")

    def _split_lora_plus(self, params, ratio):
        """LoRA+ (ICML 2024): матрице B — LR в ratio раз выше, чем A.
        До ~2x быстрее сходимость и +1-2% качества при той же цене шага.
        Возвращает список param_groups или None (выкл / нет B-матриц)."""
        if not ratio or ratio <= 1:
            return None
        names = {id(p): n for n, p in self.model.named_parameters()}
        group_a, group_b = [], []
        for p in params:
            parts = names.get(id(p), "").split(".")
            (group_b if "lora_B" in parts else group_a).append(p)
        if not group_b:
            return None
        lr = self.config.learning_rate
        print(f"[LoRA+] Дифференциальный LR: A x1 ({len(group_a)} парам.), "
              f"B x{ratio:g} ({len(group_b)} парам.)")
        return [{"params": group_a, "lr": lr},
                {"params": group_b, "lr": lr * ratio}]

    def _build_optimizer(self):
        """Экономный выбор оптимизатора: adamw — точный, но 2 состояния на параметр;
        adamw8bit — те же шаги AdamW, состояния в int8 (~75% экономии памяти);
        adafactor — почти без состояний; sgd — минимум памяти для гигантских моделей.
        + LoRA+ (--lora_plus_ratio): B-матрицам отдельный высокий LR."""
        params = [p for p in self.model.parameters() if p.requires_grad]
        kind = getattr(self.config, "optimizer", "adamw").lower()
        lr = self.config.learning_rate
        ratio = float(getattr(self.config, "lora_plus_ratio", 0) or 0)
        groups = None
        if kind in ("adamw", "adamw8bit", "sgd"):
            groups = self._split_lora_plus(params, ratio)
        elif ratio > 1:
            print("[LoRA+] Работает только с adamw/adamw8bit/sgd — для "
                  f"{kind} игнорирую.")
        P = groups if groups is not None else params
        if kind == "sgd":
            print("[Оптимизатор] SGD с momentum 0.9 (минимум памяти)")
            return torch.optim.SGD(P, lr=lr * 5, momentum=0.9)
        if kind == "adafactor":
            try:
                from transformers.optimization import Adafactor
                print("[Оптимизатор] Adafactor (без вторых моментов — для больших моделей)")
                return Adafactor(params, lr=lr, relative_step=False, warmup_init=False)
            except Exception as e:
                print(f"[Оптимизатор] Adafactor недоступен ({e}), падаю на SGD")
                return torch.optim.SGD(params, lr=lr * 5, momentum=0.9)
        if kind == "adamw8bit":
            # Если есть bitsandbytes + CUDA — берём его Paged-вариант (до +25%
            # пропускной по статье QLoRA-профилирования + точнее квантование),
            # иначе встроенный optim8bit.Adam8bit (работает везде, без зависимостей)
            try:
                if torch.cuda.is_available():
                    import bitsandbytes as bnb
                    if hasattr(bnb.optim, "PagedAdamW8bit"):
                        print("[Оптимизатор] bitsandbytes PagedAdamW8bit (8 бит + защита от всплесков)")
                        return bnb.optim.PagedAdamW8bit(P, lr=lr, weight_decay=0.01)
                    print("[Оптимизатор] bitsandbytes AdamW8bit (состояния в 8 бит)")
                    return bnb.optim.AdamW8bit(P, lr=lr, weight_decay=0.01)
                raise ImportError("нет CUDA — bitsandbytes не подойдёт")
            except Exception as e:
                from optim8bit import Adam8bit
                print(f"[Оптимизатор] Встроенный Adam8bit ({e}; состояния int8, ~75% экономии)")
                return Adam8bit(P, lr=lr, weight_decay=0.01)
        if kind == "galore":
            from galore import GaLoreAdamW
            rank = getattr(self.config, "galore_rank", 128)
            gap = getattr(self.config, "galore_gap", 200)
            scale = getattr(self.config, "galore_scale", 1.0)
            print(f"[Оптимизатор] GaLore (rank={rank}, SVD каждые {gap} шагов, scale={scale}): "
                  f"обучение 100% весов, состояния в низкоранговом подпространстве")
            return GaLoreAdamW(params, lr=lr, weight_decay=0.01,
                               rank=rank, update_gap=gap, scale=scale)
        print("[Оптимизатор] AdamW")
        return torch.optim.AdamW(P, lr=lr, weight_decay=0.01)

    def _setup_amp(self):
        """AMP: на CUDA — fp16 (тензорные ядра, до ~2x быстрее), на CPU — bf16
        (fp16 на CPU почти не поддерживается вычислительными ядрами и был бы
        медленнее fp32; bf16 даёт экономию памяти/трафика без потери скорости).
        bf16 не требует GradScaler (широкая экспонента — нет переполнений)."""
        want = getattr(self.config, "amp", True)
        pref = getattr(self.config, "amp_dtype", "auto")
        cuda = torch.cuda.is_available()
        self._amp_device = "cuda" if cuda else "cpu"
        self._scaler = None
        if not want:
            self._amp_enabled = False
            self._amp_dtype = torch.float32
            print("[AMP] выключен (--no_amp)")
            return
        if pref == "fp16":
            dtype = torch.float16
            if self._amp_device == "cpu":
                print("[AMP] fp16 на CPU не ускоряет (нет аппаратных ядер) — переключаю на bf16")
                dtype = torch.bfloat16
        elif pref == "bf16":
            dtype = torch.bfloat16
        else:  # auto
            dtype = torch.float16 if cuda else torch.bfloat16
        try:
            with torch.autocast(self._amp_device, dtype, enabled=True):
                _ = torch.ones(4, 4) @ torch.ones(4, 4)
            ok = True
        except Exception as e:
            print(f"[AMP] {dtype} не поддерживается ({e}) — AMP выключен")
            ok = False
        self._amp_enabled = ok
        self._amp_dtype = dtype if ok else torch.float32
        if ok and dtype == torch.float16 and self._amp_device == "cuda":
            try:
                self._scaler = torch.amp.GradScaler("cuda")
            except Exception:
                self._scaler = torch.cuda.amp.GradScaler()
        if ok:
            extra = " + GradScaler" if self._scaler is not None else " (без скейлера)"
            print(f"[AMP] включён: {self._amp_device}/{dtype}{extra}")

    def load_model_and_tokenizer(self):
        print(f"\n[1/3] Загрузка токенизатора...")
        tokenizer_src = self.config.model_name_or_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_src,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        print(f"[2/3] Загрузка базовой модели в память...")
        load_src = self.config.model_name_or_path
        if self._tune_mode == "full":
            ft_ckpt = os.path.join(self.config.output_dir, "interactive_checkpoint")
            if self._weights_file(ft_ckpt) is not None:
                print(f" Full-веса прошлого запуска найдены в '{ft_ckpt}' — продолжаю с них.")
                load_src = ft_ckpt
        try:
            self.model, load_info = load_model_auto(
                load_src,
                dtype_str=getattr(self.config, "dtype", "auto"),
                gradient_checkpointing=getattr(self.config, "gradient_checkpointing", True),
                force_mmap=getattr(self.config, "force_mmap", None),
                attn=getattr(self.config, "attn", "auto"),
                quant4=getattr(self.config, "quant4", "auto"),
            )
            if load_info["mmap"]:
                print(f" Модель больше RAM ({load_info['weights_gb']:.1f} ГБ) — mmap-режим, веса лежат на диске.")
            else:
                print(f" Модель ({load_info['weights_gb']:.1f} ГБ) загружена в RAM обычным способом.")
            self._model_family = self._detect_model_family()
            self._place_model(load_info)
        except Exception as e:
            print(f" Универсальная загрузка не удалась ({e}), пробуем классический путь...")
            try:
                dtype = torch.bfloat16
                self.model = AutoModelForCausalLM.from_pretrained(
                    load_src,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            except Exception:
                dtype = torch.float32
                self.model = AutoModelForCausalLM.from_pretrained(
                    load_src,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
            self._model_family = self._detect_model_family()
            self._place_model(None)

        # Liger-патчи — на чистую базу, ДО оборачивания в PEFT
        self._maybe_apply_liger()

        if self._tune_mode == "full":
            print("[3/3] Full-FT режим (без LoRA — обучаются веса базы, под GaLore)...")
            if self.resume_checkpoint and os.path.exists(
                    os.path.join(self.resume_checkpoint, "adapter_model.safetensors")):
                print("[Full-FT] LoRA-чекпоинт для resume несовместим с full-режимом — начинаю с весов базы.")
            self._setup_full_ft()
        else:
            print(f"[3/3] Подключение LoRA (PEFT) адаптеров...")
            self._setup_adapters()

        gc.collect()

        # Страховка от 96-99% RAM: если после загрузки памяти почти нет — кричим сразу
        try:
            import psutil
            if psutil.virtual_memory().percent > 85:
                print("[ПРЕДУПРЕЖДЕНИЕ] RAM занята >85% после загрузки модели!")
                print("  Останови обучение (stop) и перезапусти с флагом --mmap on и --max_length 384")
        except Exception:
            pass

    def _setup_full_ft(self):
        """Full-finetune режим для GaLore: без PEFT, размораживаем проекции
        самой базы (q/k/v/o/gate/up/down). Обучаются 100% весов активных
        слоёв (см. LISA), а не адаптеры."""
        frozen = unfrozen = 0
        for name, p in self.model.named_parameters():
            p.requires_grad = False
            frozen += 1
        for name, p in self.model.named_parameters():
            if self._is_rotated_param(name):
                p.requires_grad = True
                unfrozen += 1
        print(f"[Full-FT] База заморожена ({frozen} тензоров), разморожено проекций: {unfrozen}")
        # Gradient checkpointing без входных градиентов не сработает (как и с PEFT)
        if getattr(self.config, "gradient_checkpointing", True):
            try:
                self.model.enable_input_require_grads()
            except Exception:
                pass
        try:
            train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"[Full-FT] Обучаемых параметров: {train/1e9:.3f}B из {total/1e9:.1f}B "
                  f"({100*train/max(1,total):.2f}%)")
        except Exception:
            pass

    def _resume_adapter_dir(self):
        """Папка LoRA-чекпоинта, с которой реально продолжаем (проверка базы).
        Возвращает путь или None (старт с нуля)."""
        rp = self.resume_checkpoint
        if rp and os.path.exists(os.path.join(rp, "adapter_model.safetensors")):
            try:
                import json
                with open(os.path.join(rp, "adapter_config.json")) as f:
                    chk_cfg = json.load(f)
                if chk_cfg.get("base_model_name_or_path") == self.config.model_name_or_path:
                    return rp
                print(f"[Resume] Адаптер в '{rp}' от другой базы "
                      f"({chk_cfg.get('base_model_name_or_path')}) — старт с нуля.")
            except Exception:
                pass
        return None

    @staticmethod
    def _weights_file(weights_dir):
        """Файл весов для сигнатуры: адаптер (LoRA) или model.safetensors (full)."""
        if weights_dir is None:
            return None
        cand = os.path.join(weights_dir, "adapter_model.safetensors")
        if os.path.exists(cand):
            return cand
        cand = os.path.join(weights_dir, "model.safetensors")
        if os.path.exists(cand):
            return cand
        shards = sorted(glob.glob(os.path.join(weights_dir, "*.safetensors")))
        return shards[0] if shards else None

    @staticmethod
    def _file_sig(path):
        if path is None or not os.path.exists(path):
            return None
        st = os.stat(path)
        return [os.path.basename(path), st.st_mtime_ns, st.st_size]

    def _setup_adapters(self):
        # Verify if resume checkpoint is valid for this base model
        resume_dir = self._resume_adapter_dir()
        can_resume = resume_dir is not None

        if can_resume:
            print(f" Загрузка ранее обученных LoRA весов из '{resume_dir}' для продолжения обучения...")
            self.model = PeftModel.from_pretrained(self.model, resume_dir, is_trainable=True)
            self.model.print_trainable_parameters()
        else:
            # Универсальное обнаружение чистых torch.nn.Linear для LoRA:
            # точное совпадение последнего компонента имени (q_proj, gate_proj...),
            # мультимодальные башни (vision/audio) пропускаем.
            # Работает для Gemma, Qwen, Llama и других HF-архитектур.
            target_linear_modules = []
            for name, mod in self.model.named_modules():
                if isinstance(mod, torch.nn.Linear) or type(mod).__name__ == "Linear":
                    if name.split(".")[-1] in LORA_PROJ_NAMES:
                        if not any(hint in name.lower() for hint in VISION_AUDIO_HINTS):
                            target_linear_modules.append(name)

            if not target_linear_modules:
                target_linear_modules = self.config.lora_target_modules
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=target_linear_modules,
                bias="none"
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()

        # Gradient checkpointing + PEFT: без входных градиентов чекпоинтинг не сработает
        if getattr(self.config, "gradient_checkpointing", True):
            try:
                self.model.enable_input_require_grads()
            except Exception:
                pass

    def _interactive_command_listener(self):
        time.sleep(1)
        print("\n" + "#" * 60)
        print(" [ИНТЕРАКТИВНОЕ УПРАВЛЕНИЕ ВО ВРЕМЯ ОБУЧЕНИЯ]")
        print(" Введите команду в консоль и нажмите Enter:")
        print("   game   - режим для игр (минимальная нагрузка CPU)")
        print("   work   - режим для работы/браузера (средняя нагрузка)")
        print("   max    - режим на полную (отошел поесть/гулять)")
        print("   turbo  - задействовать 100% всех ядер")
        print("   pause  - поставить обучение на паузу")
        print("   resume - продолжить обучение")
        print("   save   - немедленно сохранить чекпоинт")
        print("   stop   - остановить и сохранить модель")
        print("#" * 60 + "\n")
        
        while not self.stop_requested:
            try:
                cmd = sys.stdin.readline()
                if not cmd:
                    break
                cmd = cmd.strip().lower()
                if not cmd:
                    continue
                
                if cmd in SPEED_PRESETS:
                    self.config.apply_preset(cmd)
                    self.res_mgr.apply_preset(SPEED_PRESETS[cmd])
                    print(f"\n>>> [РЕЖИМ ИЗМЕНЕН НА]: {SPEED_PRESETS[cmd]['name']} (Потоков: {self.res_mgr.cpu_threads}) <<<\n")
                elif cmd == "resume":
                    self.res_mgr.is_paused = False
                    print("\n>>> [ПРОДОЛЖЕНИЕ] Обучение возобновлено! <<<\n")
                elif cmd.startswith("threads "):
                    try:
                        n = int(cmd.split()[1])
                        self.res_mgr.apply_cpu_threads(n)
                        print(f"\n>>> Число потоков CPU изменено на: {n} <<<\n")
                    except ValueError:
                        print("Формат: threads <число>")
                elif cmd == "save":
                    print("\n>>> Сохранение текущего состояния адаптеров... <<<\n")
                    self.save_model("interactive_checkpoint")
                    try:
                        self._save_training_state(
                            reason="по команде save",
                            weights_dir=os.path.join(self.config.output_dir, "interactive_checkpoint"))
                    except Exception:
                        pass
                elif cmd == "pause":
                    self.res_mgr.is_paused = True
                    print("\n>>> [ПАУЗА] Обучение приостановлено. Введите 'resume' для продолжения. <<<\n")
                    print(">>> Заодно сохраняю прогресс (пригодится, если закроешь окно)... <<<\n")
                    try:
                        self.save_model("interactive_checkpoint")
                        self._save_training_state(
                            reason="на паузе",
                            weights_dir=os.path.join(self.config.output_dir, "interactive_checkpoint"))
                    except Exception as e:
                        print(f">>> Сохранить на паузе не вышло ({e}) — просто продолжай обучение. <<<")
                elif cmd == "stop":
                    print("\n>>> Запрос на завершение обучения... <<<\n")
                    self.stop_requested = True
                    break
                elif cmd == "status":
                    stats = self.res_mgr.get_stats()
                    print(f"\n>>> СТАТУС: Потоки={stats.get('active_threads')}, RAM={stats.get('ram_used_mb', 0):.1f}MB, Пауза={stats.get('is_paused')} <<<\n")
                else:
                    print(f"Неизвестная команда '{cmd}'. Доступно: game, work, max, turbo, pause, resume, save, stop, status")
            except Exception:
                break

    def _print_dataset_advisory(self):
        """Совет по LR/grad_accum исходя из размера датасета: чем больше данных,
        тем аккуратнее модель должна их впитывать."""
        try:
            size_gb = os.path.getsize(self.config.train_file) / 1e9
        except OSError:
            return
        if size_gb < 0.5:
            rec_lr, note = 3e-5, "маленький датасет"
        elif size_gb < 3.0:
            rec_lr, note = 2e-5, "средний датасет"
        else:
            rec_lr, note = 1.5e-5, "большой датасет (гигабайты): медленное впитывание"
        print(f"[Совет] Датасет {size_gb:.2f} ГБ ({note}): рекомендуемый LR={rec_lr:.1e}")
        if self.config.learning_rate > rec_lr * 2:
            print(f"[Совет] ВНИМАНИЕ: твой LR={self.config.learning_rate:.0e} в "
                  f"{self.config.learning_rate/rec_lr:.0f}x выше рекомендуемого — риск скачков лосса!")
        if size_gb > 1.0 and self.config.gradient_accumulation_steps < 8:
            print(f"[Совет] Для стабильности на большом датасете поставь --grad_accum 8 (сейчас {self.config.gradient_accumulation_steps})")

    def _maybe_compile(self):
        """torch.compile для скорости. Молча пропускаем, если нечем собрать
        (этому Windows-ПК нужен C++ компилятор; в облаке на Linux работает)."""
        if not getattr(self.config, "compile", False):
            return
        if getattr(self.config, "lisa_layers", 0) and self.config.lisa_interval < 200:
            print("[Compile] ВНИМАНИЕ: частая LISA-ротация = рекомпиляция каждый раз. "
                  "Выгода съедается; используй --lisa_interval 200+ или --lisa_layers 0.")
        try:
            self.model = torch.compile(self.model)
            print("[Compile] Модель скомпилирована (inductor). Первый шаг будет долгим (прогрев).")
        except Exception as e:
            print(f"[Compile] Не вышло ({str(e)[:200]}) — еду без компиляции.")
            self.config.compile = False

    def _place_model(self, load_info=None):
        """Куда считать: quant-модель уже разложена по GPU (device_map);
        bf16-малютку тянем на CUDA целиком; mmap-гиганта оставляем на CPU."""
        if not torch.cuda.is_available():
            return
        if (load_info or {}).get("quant"):
            print(f"[Device] Квантованная модель уже на GPU.")
            return
        try:
            if (load_info or {}).get("mmap"):
                print("[Device] CUDA есть, но модель на mmap — считаю на CPU "
                      "(для GPU нужен --quant4 on).")
                return
            self.model.to(self.device)
            print(f"[Device] Модель целиком на {self.device}.")
        except Exception as e:
            print(f"[Device] На CUDA не влезло ({str(e)[:120]}) — считаю на CPU.")

    def _maybe_apply_liger(self):
        """Liger kernels (RMSNorm/RoPE/SwiGLU/CE в одном Triton-ядре):
        +20% throughput, -60% памяти (FusedLinearCE = тот самый CutCE-эффект
        для 152k-словаря Qwen). Только Linux+CUDA+Triton; иначе пропуск."""
        if not getattr(self.config, "liger", False):
            return
        try:
            import importlib
            if importlib.util.find_spec("liger_kernel") is None:
                raise ImportError("pip install liger-kernel")
            if importlib.util.find_spec("triton") is None:
                raise ImportError("нужен Triton (Linux+CUDA)")
            from liger_kernel import transformers as LK
            mt = str(getattr(getattr(self.model, "config", None), "model_type", "") or "")
            table = {
                "llama": "apply_liger_kernel_to_llama",
                "qwen2": "apply_liger_kernel_to_qwen2",
                "qwen3": "apply_liger_kernel_to_qwen3",
                "mistral": "apply_liger_kernel_to_mistral",
                "gemma": "apply_liger_kernel_to_gemma",
                "gemma2": "apply_liger_kernel_to_gemma2",
                "phi3": "apply_liger_kernel_to_phi3",
            }
            key = mt.lower().replace("-", "_")
            cands = [key]
            # qwen3_5 -> qwen3, llama4 -> llama и т.п.
            stem = key.rstrip("0123456789_")
            if stem != key:
                cands.append(stem)
            fn = next((getattr(LK, table[c], None) for c in cands if c in table), None)
            if fn is None:
                print(f"[Liger] Нет патчера для model_type '{mt}' — пропускаю.")
                self.config.liger = False
                return
            fn(self.model)
            print(f"[Liger] Патчи применены ({mt}).")
        except Exception as e:
            print(f"[Liger] Пропуск ({e}) — еду на обычных ядрах.")
            self.config.liger = False

    def _count_samples(self):
        """Реальное число строк в train-файле (быстрый подсчёт по \\n).
        Раньше здесь стояло фиктивное 10000 — из-за чего линейный шедулер
        загонял LR в ноль на 312-м шаге (accum 32), и ночи умирали, не
        доучив датасет. Теперь горизонт = реальный объём."""
        if self.config.max_samples:
            return int(self.config.max_samples)
        path = self.config.train_file
        try:
            if str(path).lower().endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                    return max(1, sum(1 for _ in f))
            n = 0
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    n += chunk.count(b"\n")
            return max(1, n)
        except Exception:
            return 10000

    def _dataset_line(self, ds):
        """Номер строки файла, где датасет сейчас (для мгновенного resume)."""
        d = getattr(ds, "main", ds)
        return int(getattr(d, "current_line", 0) or 0)

    def _estimate_avg_doc_len(self, n: int = 128) -> float:
        """Средняя длина диалога в токенах.

        Нужна для горизонта LR при packing: в окно влезает несколько диалогов,
        значит шагов на эпоху меньше, чем сэмплов. Замер идёт на отдельном
        экземпляре датасета, RNG восстанавливаем — порядок сэмплов в
        основном потоке от замера не сдвигается.
        """
        cached = getattr(self, "_avg_doc_len", None)
        if cached:
            return cached
        from dataset_formatter import StreamingConversationDataset
        rng_state = random.getstate()
        try:
            probe = StreamingConversationDataset(
                jsonl_path=self.config.train_file,
                tokenizer=self.tokenizer,
                identity_path="",
                identity_mix_ratio=0.0,
                max_length=self.config.max_length,
                max_samples=n,
                train_only_on_assistant=self.config.train_only_on_assistant,
                cache_dir=getattr(self.config, "cache_dir", None),
            )
            lengths = [s["input_ids"].size(0) for s in probe]
        except Exception as e:
            print(f"[Шедулер] Не смог измерить среднюю длину диалога ({str(e)[:60]}) — беру max_length.")
            lengths = []
        finally:
            random.setstate(rng_state)
        value = (sum(lengths) / len(lengths)) if lengths else float(self.config.max_length)
        self._avg_doc_len = value
        return value

    def _build_scheduler(self, optimizer):
        """Линейный шедулер с разогревом (долей от всего обучения).
        Горизонт — реальное число шагов на эпоху, иначе LR обнулится раньше
        конца датасета."""
        samples = self._count_samples()
        # +15% подмес identity-репетиции
        total_samples = int(samples * 1.15) * self.config.num_epochs
        estimated_steps = max(1, total_samples // self.config.gradient_accumulation_steps)
        if getattr(self.config, "packing", False):
            # При packing один шаг — это окно токенов, а не один диалог:
            # шагов на эпоху в (max_length / средняя длина) раз меньше.
            avg_len = self._estimate_avg_doc_len()
            per_step = max(1, self.config.max_length * max(1, self.config.batch_size)
                           * self.config.gradient_accumulation_steps)
            estimated_steps = max(1, int(total_samples * avg_len // per_step))
            print(f"[Шедулер] Packing: средний диалог ~{avg_len:.0f} токенов, "
                  f"в окно {self.config.max_length} влезает ~"
                  f"{self.config.max_length / max(1.0, avg_len):.1f} диалогов")
        if getattr(self.config, "warmup_ratio", 0) > 0:
            warmup_steps = max(1, int(estimated_steps * self.config.warmup_ratio))
        else:
            warmup_steps = self.config.warmup_steps
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max(1, estimated_steps)
        )
        print(f"[Шедулер] Сэмплов на эпоху: ~{int(samples * 1.15)} (файл {samples} + 15% identity) "
              f"-> шагов ~{estimated_steps}, warmup {warmup_steps}. "
              f"LR обнулится на последнем шаге эпохи, а не раньше.")
        return lr_scheduler, warmup_steps

    def _start_listener(self):
        listener_thread = threading.Thread(target=self._interactive_command_listener, daemon=True)
        listener_thread.start()
        return listener_thread

    def _init_lisa_and_guards(self, resume_active=None):
        """LISA-инициализация + страж памяти логитов. Общая для SFT и ORPO.
        resume_active: сохранённый набор слоёв — применяем напрямую без
        случайной ротации (иначе съедет RNG-последовательность)."""
        # LISA-инициализация: работаем только с N случайными слоями,
        # остальные обучаемые параметры заморожены (не тратят память/время)
        if getattr(self.config, "lisa_layers", 0) and self.config.lisa_layers > 0:
            self._collect_lisa_groups()
            n_layers = len([k for k in self._lisa_groups if k >= 0])
            print(f"[LISA] Послойное обучение: {self.config.lisa_layers} из {n_layers} слоёв, "
                  f"ротация каждые {self.config.lisa_interval} шагов")
            if resume_active is not None:
                frozen, unfrozen = self._apply_lisa(set(resume_active))
                self._lisa_active = sorted(set(resume_active))
                print(f"[LISA] Восстановлены активные слои с прошлого запуска: {self._lisa_active} "
                      f"(заморожено: {frozen}, активно: {unfrozen})")
            else:
                self._rotate_lisa(0)
            if self.config.kl_anchor > 0:
                print("[LISA] ВНИМАНИЕ: KL-якорь делает второй форвард базы — "
                      "на гигантской модели будет ~2x медленнее. При желании выключи (--kl_anchor 0).")

        # Страж памяти логитов: vocab * seq может съесть гигабайты на 120B-моделях.
        # В ORPO два форварда (chosen+rejected) — запас считаем с коэффициентом 2.
        try:
            vs = getattr(getattr(self.model, "config", None), "vocab_size", 0) or 0
            if vs == 0:
                vs = getattr(getattr(self.model, "config", None), "text_config", None) is not None and \
                    getattr(self.model.config.text_config, "vocab_size", 0) or 0
            if vs:
                mult = 2 if getattr(self.config, "objective", "sft") == "orpo" else 1
                est_gb = mult * vs * self.config.max_length * 4 / 1e9
                if est_gb > 1.0:
                    print(f"[Память] Логиты ~{est_gb:.1f} ГБ (vocab {vs} x seq {self.config.max_length}"
                          f"{' x2 (ORPO)' if mult == 2 else ''}): "
                          f"если будет OOM — уменьши --max_length (сейчас {self.config.max_length}).")
        except Exception:
            pass

    def _optim_step(self, optimizer, lr_scheduler):
        """Клиппинг + шаг + шедулер с учётом GradScaler (fp16 на CUDA)."""
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
        if self._scaler is not None:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    STATE_FILENAME = "training_state.pt"

    def _resolve_state_file(self):
        """Путь файла состояния. resume_state: 'auto' (по умолчанию,
        <output_dir>/training_state/), путь к файлу/папке или 'off'.
        ВАЖНО: 'off' означает только «НЕ ЗАГРУЖАТЬ» — запись продолжается
        (иначе ночь не сможет продолжиться дальше)."""
        mode = getattr(self.config, "resume_state", "auto")
        if mode and mode != "auto" and mode != "off":
            if os.path.isdir(mode):
                return os.path.join(mode, self.STATE_FILENAME)
            if mode.endswith(".pt"):
                return mode
            return os.path.join(mode, self.STATE_FILENAME)
        return os.path.join(self.config.output_dir, "training_state", self.STATE_FILENAME)

    def _trainable_names(self):
        # ВАЖНО: сравниваем РОТИРУЕМЫЙ набор (все LoRA / все проекции базы),
        # а не текущий requires_grad: LISA-ротация меняет requires_grad каждый
        # интервал, и иначе любое продолжение с LISA отклонялось бы как
        # «набор изменился» (это был живой баг: resume не работал никогда).
        return sorted(n for n, _ in self.model.named_parameters()
                      if self._is_rotated_param(n))

    def _rotatable_names(self):
        return self._trainable_names()

    def _apply_subset_optimizer_state(self, optimizer, pending) -> bool:
        """Частичное восстановление моментов: снапшот покрывает лишь часть
        ротируемых параметров (старый формат или уже ротация). Перекладка —
        позиционная (порядок обхода детерминирован), значения как есть."""
        try:
            saved = pending["optimizer"]
            if len(saved.get("param_groups", [])) != 1:
                return False
            saved_ids = list(saved["param_groups"][0]["params"])
            # Порядок параметров в optimizer = обход requires_grad на момент создания
            # (до первой LISA-ротации — т.е. ВСЕ ротируемые). Обход детерминирован,
            # поэтому saved_ids[i] соответствует i-му ротируемому имени текущего обхода.
            # Так подхватываются моменты ВСЕХ слоёв, успевших поучаствовать в ротациях
            # за 100 шагов, а не только последнего. Имена — лишь ворота (см. _load).
            cur_order = [n for n, _ in self.model.named_parameters()
                         if self._is_rotated_param(n)]
            if len(saved_ids) != len(cur_order):
                return False
            by_name = dict(self.model.named_parameters())
            restored = 0
            for idx, name in zip(saved_ids, cur_order):
                if idx not in saved.get("state", {}):
                    continue
                p = by_name[name]
                entry = saved["state"][idx]
                fresh = {}
                for k, v in entry.items():
                    fresh[k] = (v.detach().clone().to(p.device)
                                if isinstance(v, torch.Tensor) else v)
                # adam-моменты обязаны быть размером с параметр, иначе — чужой конфиг
                for k in ("exp_avg", "exp_avg_sq"):
                    if k in fresh and isinstance(fresh[k], torch.Tensor):
                        if fresh[k].numel() != p.numel():
                            return False
                optimizer.state[p] = fresh
                restored += 1
            print(f"[Продолжение] Восстановлены моменты {restored} параметров "
                  f"из {len(cur_order)}.")
            return restored > 0
        except Exception:
            return False

    def _latest_checkpoint_step(self):
        """Номер шага из последней папки checkpoint-step-N (для восстановления)."""
        try:
            best = 0
            for p in glob.glob(os.path.join(self.config.output_dir, "checkpoint-step-*")):
                tail = os.path.basename(p).rsplit("-", 1)[-1]
                if tail.isdigit():
                    best = max(best, int(tail))
            return best or None
        except Exception:
            return None

    def _apply_pending_state(self, optimizer, lr_scheduler, pending):
        """Применяет снапшот к свежесозданным optimizer/scheduler.
        Возвращает (optimizer, lr_scheduler, global_step, pending|None).
        Главное правило: НЕ откатываемся на шаг 1. Если моменты не легли —
        ведём шедулер вперёд до сохранённого шага и продолжаем обучение."""
        step = int(pending.get("global_step") or 0)

        if pending.get("skip_optimizer"):
            for _ in range(step):
                lr_scheduler.step()
            print(f"[Продолжение] Шаг {step} принят без моментов Adam "
                  f"(LR={lr_scheduler.get_last_lr()[0]:.2e}).")
            return optimizer, lr_scheduler, step, pending

        try:
            lr_scheduler.load_state_dict(pending["scheduler"])
            if pending.get("allow_subset"):
                if not self._apply_subset_optimizer_state(optimizer, pending):
                    raise RuntimeError("частичные моменты не легли")
            else:
                optimizer.load_state_dict(pending["optimizer"])
            return optimizer, lr_scheduler, step, pending
        except Exception as e:
            # Моменты не встали (другое число групп/LoRA+/иной оптимизатор) —
            # сохраняем прогресс: шаг и LR есть, моменты начнутся заново.
            print(f"[Продолжение] Моменты не применились ({e}). Продолжаю с шага {step} "
                  f"со свежими моментами Adam (прогресс НЕ теряется).")
            fresh = self._build_optimizer()
            fresh_sched, _ = self._build_scheduler(fresh)
            for _ in range(step):
                fresh_sched.step()
            self._opt, self._sched = fresh, fresh_sched
            pending = dict(pending)
            pending["skip_optimizer"] = True
            return fresh, fresh_sched, step, pending

    def _save_training_state(self, reason="", weights_dir=None):
        """Снапшот для ТОЧНОГО продолжения: шаг, эпоха, позиция в данных,
        оптимизатор, шедулер, LISA-слои, RNG. Зовётся на чекпоинтах и в финале.
        weights_dir — папка только что сохранённых весов: пишем её сигнатуру,
        чтобы при загрузке убедиться, что моменты соответствуют ЭТИМ весам."""
        path = self._resolve_state_file()
        if path is None or self._opt is None or self._sched is None:
            return
        if not getattr(self.config, "state_save", True):
            return
        try:
            replay_ref = getattr(self, "_replay_ref", None)
            wf = self._weights_file(weights_dir)
            data = {
                "version": 1,
                "global_step": self._global_step,
                "epoch": self._cur_epoch,
                "epoch_batches": self._cur_epoch_batches,
                "line": self._cur_line,
                "optimizer": self._opt.state_dict(),
                "scheduler": self._sched.state_dict(),
                "tune_mode": self._tune_mode,
                "optimizer_name": getattr(self.config, "optimizer", "adamw"),
                "trainable_names": self._trainable_names(),
                "lisa_active": list(getattr(self, "_lisa_active", []) or []),
                "python_random_state": random.getstate(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "replay_rng_state": replay_ref.get_rng_state() if replay_ref is not None else None,
                "weights_sig": self._file_sig(wf),
                "lr": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "train_file": os.path.abspath(self.config.train_file),
                "kd": bool(getattr(self.config, "kd", False)),
                "kd_weight": float(getattr(self.config, "kd_weight", 0.0)),
                "kd_top_k": int(getattr(self.config, "kd_top_k", 0)),
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Атомарно + с fsync: иначе перезагрузка/сброс питания оставляет
            # обрезанный файл (именно это случилось 29.10 в 10:41).
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                torch.save(data, f)
                f.flush()
                os.fsync(f.fileno())
            # Прошлую версию держим как .prev — страховка на случай порчи
            if os.path.exists(path):
                try:
                    os.replace(path, path + ".prev")
                except Exception:
                    pass
            os.replace(tmp, path)
            print(f"[Состояние] Шаг {self._global_step} (эпоха {self._cur_epoch}, "
                  f"батчей {self._cur_epoch_batches}) сохранён {reason}-> '{path}'")
        except Exception as e:
            print(f"[Состояние] Не удалось сохранить ({e}) — продолжение будет с весов.")
            try:
                if os.path.exists(path + ".tmp"):
                    os.remove(path + ".tmp")
            except Exception:
                pass

    def _load_training_state(self):
        """Читает снапшот и проверяет совместимость. Возвращает dict или None.
        RNG восстанавливает сразу; остальное применяется позже (нужны optimizer/scheduler)."""
        mode = getattr(self.config, "resume_state", "auto")
        if mode == "off":
            p = self._resolve_state_file()
            extra = ""
            if p and os.path.exists(p):
                try:
                    d = torch.load(p, map_location="cpu", weights_only=False)
                    extra = (f" ВНИМАНИЕ: на диске ЕСТЬ состояние на шаге "
                             f"{d.get('global_step')} — ты его игнорируешь и начнёшь с шага 1.")
                except Exception:
                    pass
            print("[Продолжение] Загрузка состояния отключена (--no_resume_state) — "
                  "стартуем с нуля, но состояние продолжит записываться." + extra)
            return None
        path = self._resolve_state_file()
        if path is None:
            return None
        # Основной файл мог обрезаться при сбросе питания — пробуем .prev
        candidates = [path, path + ".prev"]
        data = None
        err = None
        for cand in candidates:
            if not os.path.exists(cand):
                continue
            try:
                data = torch.load(cand, map_location="cpu", weights_only=False)
                if cand != path:
                    print(f"[Продолжение] Основной state повреждён — взял резервную копию "
                          f"'{os.path.basename(cand)}' (шаг {data.get('global_step')}).")
                break
            except Exception as e:
                err = e
                data = None
        if data is None:
            latest = self._latest_checkpoint_step()
            hint = (f"\n  Восстановление: --resume_step {latest} --resume "
                    f"{os.path.join(self.config.output_dir, 'interactive_checkpoint')}"
                    if latest else "")
            if os.path.exists(path) or os.path.exists(path + ".prev"):
                print(f"[Продолжение] Битый файл состояния ({err}).{hint}")
            else:
                print(f"[Продолжение] Состояния пока нет ({path}) — первый запуск, старт с нуля.")
            return None
        problems = []
        if data.get("tune_mode") != self._tune_mode:
            problems.append(f"tune_mode: было {data.get('tune_mode')}, сейчас {self._tune_mode}")
        if data.get("optimizer_name") != getattr(self.config, "optimizer", "adamw"):
            problems.append(f"optimizer: был {data.get('optimizer_name')}, сейчас {self.config.optimizer}")
        if data.get("trainable_names") != self._trainable_names():
            saved_set = set(data.get("trainable_names") or [])
            cur_set = set(self._rotatable_names())
            # Мягкий случай: снапшот покрывает ПОДМНОЖЕСТВО ротируемых
            # (записан до фикса / уже ротация) + веса те же + тот же оптимизатор:
            # моменты подхватим частично, остальное стартует свежим.
            if (len(problems) == 0 and saved_set and saved_set <= cur_set
                    and data.get("optimizer_name") == getattr(self.config, "optimizer", "adamw")
                    and data.get("tune_mode") == self._tune_mode):
                data["allow_subset"] = True
                print("[Продолжение] Снапшот покрывает часть параметров — "
                      "моменту подхватят совпадающие, остальные начнут свежими.")
            else:
                problems.append("набор обучаемых параметров другой (слои/LoRA/ранг изменились?)")
        # Спаривание с весами: моменты оптимизатора верны только для тех весов,
        # рядом с которыми лежал state. Свежие веса + старые моменты = тихая порча.
        if self._tune_mode == "full":
            ft_dir = os.path.join(self.config.output_dir, "interactive_checkpoint")
            cur_sig = self._file_sig(self._weights_file(ft_dir))
            if cur_sig is None:
                problems.append("full-веса прошлого запуска не найдены "
                                "(interactive_checkpoint пуст — старт с весов базы)")
            elif data.get("weights_sig") != cur_sig:
                problems.append("full-веса новее/иные, чем в момент снапшота")
        else:
            resume_dir = self._resume_adapter_dir()
            if resume_dir is None:
                problems.append("LoRA-веса прошлого запуска не найдены "
                                "(стартуем со свежего адаптера — моменты неприменимы)")
            elif data.get("weights_sig") != self._file_sig(self._weights_file(resume_dir)):
                problems.append("LoRA-веса не совпадают со снапшотом (другая запись?)")
        if problems:
            # НИКОГДА не откатываемся на 1-й шаг из-за несовместимых моментов.
            # Теряем только состояния Adam (догонят за пару сотен шагов),
            # а шаг, LR, позиция в файле и LISA-слои остаются.
            print("[Продолжение] Моменты оптимизатора неприменимы ("
                  + "; ".join(problems) + ")")
            print(f"[Продолжение] Шаг {data.get('global_step')}, строка {data.get('line')}, "
                  f"LR-шедулер — сохраняются. Adam начнёт моменты заново.")
            data["skip_optimizer"] = True
            data["problems"] = problems
        if abs(float(data.get("lr", 0)) - float(self.config.learning_rate)) > 1e-12:
            print(f"[Продолжение] ВНИМАНИЕ: LR отличается (было {data.get('lr')}, "
                  f"сейчас {self.config.learning_rate}) — шедулер продолжит со старого шага.")
        if data.get("gradient_accumulation_steps") != self.config.gradient_accumulation_steps:
            print(f"[Продолжение] ВНИМАНИЕ: grad_accum отличается "
                  f"(было {data.get('gradient_accumulation_steps')}, "
                  f"сейчас {self.config.gradient_accumulation_steps}) — шаги станут "
                  f"длиннее/короче, позиция в данных в батчах сохраняется.")
        if data.get("batch_size") != self.config.batch_size:
            print(f"[Продолжение] ВНИМАНИЕ: batch_size отличается "
                  f"(было {data.get('batch_size')}, сейчас {self.config.batch_size}).")
        if bool(data.get("kd", False)) != bool(getattr(self.config, "kd", False)):
            print("[Продолжение] ВНИМАНИЕ: режим KD отличается "
                  f"(было {bool(data.get('kd', False))}, сейчас {bool(getattr(self.config, 'kd', False))}).")
        elif getattr(self.config, "kd", False) and abs(
                float(data.get("kd_weight", 0.0)) - float(self.config.kd_weight)) > 1e-9:
            print(f"[Продолжение] ВНИМАНИЕ: вес KD отличается "
                  f"(было {data.get('kd_weight')}, сейчас {self.config.kd_weight}).")
        if data.get("train_file") and os.path.abspath(self.config.train_file) != data["train_file"]:
            print(f"[Продолжение] ВНИМАНИЕ: файл датасета другой!\n"
                  f"  был: {data['train_file']}\n"
                  f"  сейчас: {os.path.abspath(self.config.train_file)}\n"
                  f"  Позиция в данных (шаг/батчи) относится к СТАРОМУ файлу — "
                  f"лучше начать этап заново (--no_resume_state).")
        try:
            random.setstate(data["python_random_state"])
            torch.set_rng_state(data["torch_random_state"])
            if data.get("cuda_random_state") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(data["cuda_random_state"])
        except Exception as e:
            print(f"[Продолжение] Не удалось восстановить RNG ({e}) — порядок сэмплов может отличаться.")
        return data

    def train(self):
        self.load_model_and_tokenizer()

        # ORPO: SFT + предпочтения одним этапом (нужны пары chosen/rejected)
        if getattr(self.config, "objective", "sft") == "orpo":
            if getattr(self.config, "kd", False):
                print("[KD] Дистилляция несовместима с ORPO — отключаю KD.")
                self.config.kd = False
            self._train_orpo()
            return

        if getattr(self.config, "kd", False):
            if not _file_has_soft(self.config.train_file):
                print("[KD] В датасете нет поля soft — дистилляция отключена. "
                      "Нужен файл, созданный run_distill.py с шансами (train_distill.kd.jsonl.gz).")
                self.config.kd = False
            elif getattr(self.config, "packing", False):
                print("[KD] Packing несовместим с soft-targets — отключаю packing.")
                self.config.packing = False
            else:
                print(f"[KD] Дистилляция включена: вес {self.config.kd_weight:g}, "
                      f"top-k {self.config.kd_top_k}.")
                if self.config.max_length < 512:
                    print(f"[KD] ВНИМАНИЕ: контекст {self.config.max_length} — большинство ответов "
                          f"не влезет и будет пропущено. Освободи RAM или подними контекст.")
                elif self.config.max_length < 1024:
                    print(f"[KD] Контекст {self.config.max_length}: ответы длиннее будут пропущены; "
                          f"для стабильного KD лучше 1024+ или короткие ответы.")

        # Точное продолжение: шаг, эпоха, позиция в данных, оптимизатор, шедулер, RNG.
        # Применяется после создания optimizer/scheduler (см. ниже).
        pending = self._load_training_state()
        start_epoch = 0
        skip_batches = 0

        # Разбираем --resume_step ДО создания датасета: тогда скип будет
        # мгновенным (старт с нужной строки), а не через перебор батчей.
        rs = getattr(self.config, "resume_step", None)
        if isinstance(rs, str) and rs.lower() == "auto":
            rs = self._latest_checkpoint_step()
            print(f"[Продолжение] --resume_step auto -> шаг {rs} (из последней папки чекпоинта).")
        if rs and pending is not None:
            print(f"[Продолжение] --resume_step {rs} игнорирую: найдено состояние "
                  f"на шаге {pending.get('global_step')} (оно свежее).")
            rs = None
        self._rs = int(rs) if rs else 0
        est_line = int(pending.get("line") or 0) if pending else 0
        if not est_line and self._rs:
            # Один шаг = accum сэмплов; с identity-подмесом сэмплов на 15% больше строк
            est_line = int(self._rs * self.config.gradient_accumulation_steps / 1.15)
            print(f"[Продолжение] --resume_step {self._rs}: стартую примерно со строки "
                  f"{est_line} (скип мгновенный).")

        # --- Решение по padding: что вообще включаем ---
        # Считается ДО сборки датасета, иначе отключить уже применённую
        # обёртку поздно.
        if getattr(self.config, "packing", False):
            avg_len = self._estimate_avg_doc_len()
            if avg_len > 0.9 * self.config.max_length:
                # Диалоги почти все обрезаны до max_length — в окно влезает
                # ровно один диалог, упаковывать нечего, а блочная маска
                # только замедляет форвард.
                print(f"[Padding] Диалоги обрезаны до max_length (средний {avg_len:.0f} из "
                      f"{self.config.max_length} токенов) — в окно влезает один диалог, "
                      f"упаковывать нечего. Packing отключаю.")
                self.config.packing = False
            elif getattr(self.config, "group_by_length", False):
                print("[Padding] Length grouping при packing отключаю — "
                      "packing и так убирает паддинг.")
                self.config.group_by_length = False

        train_dataset = StreamingConversationDataset(
            jsonl_path=self.config.train_file,
            tokenizer=self.tokenizer,
            identity_path=self.config.identity_file,
            identity_mix_ratio=0.15,
            max_length=self.config.max_length,
            max_samples=self.config.max_samples,
            train_only_on_assistant=self.config.train_only_on_assistant,
            cache_dir=getattr(self.config, "cache_dir", None),
            start_line=est_line,
            kd=getattr(self.config, "kd", False),
            kd_top_k=getattr(self.config, "kd_top_k", 8),
        )

        # Length grouping: сэмплы сортируются по длине внутри окна — в батче
        # нет разброса длин, паддинга почти не остаётся.
        if getattr(self.config, "group_by_length", False):
            window = getattr(self.config, "length_window", 64)
            train_dataset = LengthGroupedDataset(train_dataset, window=window)
            print(f"[Padding] Length grouping: сортировка по длине, окно {window} сэмплов")

        # Anti-forgetting replay: подмешиваем общий датасет, чтобы модель
        # продолжала репетировать общие знания во время узкого дообучения
        if self.config.replay_file and os.path.exists(self.config.replay_file) and self.config.replay_ratio > 0:
            replay_dataset = StreamingConversationDataset(
                jsonl_path=self.config.replay_file,
                tokenizer=self.tokenizer,
                # "" вместо None: os.path.exists(None) бросает TypeError
                # и датасет репетиции вообще не создавался.
                identity_path="",
                identity_mix_ratio=0.0,
                max_length=self.config.max_length,
                max_samples=None,
                train_only_on_assistant=self.config.train_only_on_assistant,
                cache_dir=getattr(self.config, "cache_dir", None),
            )
            if getattr(self.config, "group_by_length", False):
                replay_dataset = LengthGroupedDataset(
                    replay_dataset, window=getattr(self.config, "length_window", 64))
            train_dataset = ReplayMixedDataset(
                train_dataset, replay_dataset,
                replay_ratio=self.config.replay_ratio
            )
            self._replay_ref = train_dataset
            if pending is not None and pending.get("replay_rng_state") is not None:
                try:
                    train_dataset.set_rng_state(pending["replay_rng_state"])
                except Exception as e:
                    print(f"[Продолжение] replay-RNG не восстановлен ({e}).")
            print(f"[Anti-Forget] Репетиция включена: {self.config.replay_file} с долей {self.config.replay_ratio:.0%}")
        if self.config.kl_anchor > 0:
            print(f"[Anti-Forget] KL-якорь к базовой модели включен: weight={self.config.kl_anchor} "
                  f"(шаг будет медленнее ~2x)")

        # Packing: диалоги склеиваются в окно max_length. Применяется СВЕРХУ
        # всего потока (включая репетицию), иначе в батче смешались бы
        # упакованные окна и обычные сэмплы разной длины.
        if getattr(self.config, "packing", False):
            train_dataset = PackedDataset(
                train_dataset,
                max_length=self.config.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            print(f"[Padding] Packing: до {self.config.max_length} токенов в окне, "
                  f"между документами блочная маска внимания")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=lambda b: collate_fn(b, pad_token_id=self.tokenizer.pad_token_id)
        )

        optimizer = self._build_optimizer()

        lr_scheduler, warmup_steps = self._build_scheduler(optimizer)

        self._opt, self._sched = optimizer, lr_scheduler

        global_step = 0
        if pending is not None:
            # Порядок важен: сначала шедулер, потом оптимизатор (см. torch docs).
            # При отказе — чистый старт (свежие optimizer/scheduler уже созданы).
            optimizer, lr_scheduler, global_step, pending = self._apply_pending_state(
                optimizer, lr_scheduler, pending)
            self._opt, self._sched = optimizer, lr_scheduler
            if pending is not None:
                start_epoch = pending["epoch"]
                # Продолжение по НОМЕРУ СТРОКИ файла (мгновенно), а не по батчам.
                # Старые состояния без "line" — падаем на старый скип по батчам.
                if pending.get("line"):
                    self._cur_line = int(pending["line"])
                    skip_batches = 0
                    print(f"[Продолжение] Начинаю с шага {global_step} "
                          f"(эпоха {start_epoch}, строка файла {self._cur_line} — скип мгновенный).")
                elif pending.get("epoch_batches"):
                    skip_batches = pending["epoch_batches"]
                    print(f"[Продолжение] Начинаю с шага {global_step} "
                          f"(эпоха {start_epoch}, пропускаю {skip_batches} пройденных батчей).")
                else:
                    skip_batches = 0
                    print(f"[Продолжение] Начинаю с шага {global_step} (эпоха {start_epoch}).")

        # Аварийное продолжение без state (файл потерян/битый/после ребута):
        # веса загружены, шаг/шедулер выставляем вручную, моменты Adam — заново.
        if self._rs:
            rs = self._rs
            for _ in range(rs):
                lr_scheduler.step()
            global_step = rs
            start_epoch = 0
            skip_batches = 0          # позицию уже задали через start_line
            print(f"[Продолжение] Режим --resume_step: шаг {rs}, "
                  f"LR={lr_scheduler.get_last_lr()[0]:.2e}. Моменты Adam — заново.")
        self._global_step = global_step

        self._start_listener()

        print(f"\n=== НАЧАЛО ОБУЧЕНИЯ ===")
        print(f"Всего эпох: {self.config.num_epochs}")
        print(f"Безопасный старт: LR={self.config.learning_rate:.0e} | max_grad_norm={self.config.max_grad_norm} | "
              f"grad_accum={self.config.gradient_accumulation_steps} | warmup={warmup_steps} шагов "
              f"(ratio={self.config.warmup_ratio})")
        self._print_dataset_advisory()
        print("=" * 60)

        self.model.train()
        self._maybe_compile()
        self._setup_amp()
        self._init_lisa_and_guards(
            resume_active=pending.get("lisa_active") if pending else None)

        running_loss = 0.0
        start_time = time.time()

        for epoch in range(start_epoch, self.config.num_epochs):
            if self.stop_requested:
                break

            print(f"\n--- Эпоха {epoch + 1}/{self.config.num_epochs} ---")
            self._cur_epoch = epoch
            self._cur_epoch_batches = 0
            step = 0
            iterator = iter(train_loader)
            if epoch == start_epoch and skip_batches > 0:
                print(f"[Продолжение] Пропускаю {skip_batches} уже пройденных батчей эпохи...")
                for i in range(skip_batches):
                    try:
                        next(iterator)
                    except StopIteration:
                        break
                    if (i + 1) % 100 == 0 or (i + 1) == skip_batches:
                        print(f"[Продолжение] Пропущено {i + 1}/{skip_batches}...")
            for batch in iterator:
                if self.stop_requested:
                    break
                self._cur_epoch_batches += 1
                self._cur_line = self._dataset_line(train_dataset)
                self.res_mgr.throttle_step()

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Packing: блочная 4D-маска между документами заменяет обычную
                # 2D-маску, а позиции сбрасываются на каждом документе.
                model_kwargs = {}
                if "position_ids" in batch:
                    model_kwargs["position_ids"] = batch["position_ids"].to(self.device)
                if "block_mask" in batch:
                    # (B, L, L) -> (B, 1, L, L): блок-диагональная маска,
                    # которую модель принимает как готовую 4D-маску.
                    attention_mask = batch["block_mask"].unsqueeze(1).to(self.device)

                # AMP: форвард в половинной точности (fp16 на CUDA, bf16 на CPU).
                # LoRA-адаптеры остаются в fp32 — точность шагов не страдает.
                with torch.autocast(self._amp_device, self._amp_dtype,
                                    enabled=self._amp_enabled):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        **model_kwargs
                    )
                
                loss = outputs.loss
                if loss is None or torch.isnan(loss):
                    continue

                # KL-якорь: не даём адаптеру уехать далеко от базовой модели.
                # Форвард базы считаем БЕЗ градиентов, адаптер на время отключён (PEFT disable_adapter).
                # В full-режиме адаптера нет — якорь неприменим.
                if self.config.kl_anchor > 0 and self._tune_mode != "full":
                    with torch.no_grad(), self.model.disable_adapter():
                        base_logits = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            **model_kwargs
                        ).logits[:, :-1]
                    student_logits = outputs.logits[:, :-1]
                    shift_labels = labels[:, 1:]
                    mask = (shift_labels != -100)
                    _dev = student_logits.device
                    kl_sum = torch.zeros((), dtype=torch.float32, device=_dev)
                    mask_cnt = torch.zeros((), dtype=torch.float32, device=_dev)
                    chunk = 32
                    for i in range(0, base_logits.shape[1], chunk):
                        m = mask[:, i:i + chunk]
                        if not m.any():
                            continue
                        bl = base_logits[:, i:i + chunk].float()
                        sl = student_logits[:, i:i + chunk].float()
                        log_pb = torch.log_softmax(bl, dim=-1)
                        pb = log_pb.exp()
                        log_ps = torch.log_softmax(sl, dim=-1)
                        kl = (pb * (log_pb - log_ps)).sum(-1)
                        kl_sum = kl_sum + (kl * m).sum()
                        mask_cnt = mask_cnt + m.sum()
                    if mask_cnt.item() > 0:
                        loss = loss + self.config.kl_anchor * (kl_sum / mask_cnt.clamp(min=1))
                    del base_logits

                if getattr(self.config, "kd", False) and "soft_mask" in batch:
                    kd_val = kd_loss_from_logits(
                        outputs.logits,
                        batch["soft_ids"], batch["soft_probs"], batch["soft_mask"],
                        labels=labels,
                    )
                    if kd_val is not None and torch.isfinite(kd_val):
                        loss = loss + self.config.kd_weight * kd_val
                        self._kd_last = float(kd_val.detach().cpu())

                loss = loss / self.config.gradient_accumulation_steps
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()
                running_loss += loss.item() * self.config.gradient_accumulation_steps

                step += 1
                if step % self.config.gradient_accumulation_steps == 0:
                    self._optim_step(optimizer, lr_scheduler)
                    global_step += 1
                    self._global_step = global_step

                    if getattr(self.config, "lisa_layers", 0) and self.config.lisa_layers > 0 \
                            and global_step % self.config.lisa_interval == 0:
                        self._rotate_lisa(global_step)

                    if global_step % self.config.logging_steps == 0:
                        # Средний лосс на БАТЧ: с последнего лога прошло
                        # logging_steps optimizer-шагов по accum батчей каждый.
                        avg_loss = running_loss / (
                            self.config.logging_steps
                            * self.config.gradient_accumulation_steps
                        )
                        stats = self.res_mgr.get_stats()
                        kd_info = ""
                        if getattr(self.config, "kd", False) and self._kd_last is not None:
                            kd_info = f" | KD: {self._kd_last:.4f}"
                        print(
                            f"[Шаг {global_step}] "
                            f"Loss: {avg_loss:.4f}{kd_info} | "
                            f"LR: {lr_scheduler.get_last_lr()[0]:.2e} | "
                            f"RAM процесса: {stats.get('ram_used_mb', 0):.0f}MB | "
                            f"Потоков: {stats.get('active_threads')} | "
                            f"Профиль: {self.config.preset}"
                        )
                        running_loss = 0.0

                    if global_step % self.config.save_steps == 0:
                        self.save_model(f"checkpoint-step-{global_step}")
                        self.save_model("interactive_checkpoint")
                        self._save_training_state(
                            reason="на чекпоинте",
                            weights_dir=os.path.join(self.config.output_dir, "interactive_checkpoint"))
                        gc.collect()

        print("\n" + "=" * 60)
        print(" Обучение завершено или остановлено.")
        self.save_model("final_model")
        # Финал дублируем в interactive_checkpoint: именно оттуда стартует
        # следующий запуск (иначе продолжение взяло бы более старые веса).
        self.save_model("interactive_checkpoint")
        self._save_training_state(
            reason="в финале",
            weights_dir=os.path.join(self.config.output_dir, "interactive_checkpoint"))
        print("=" * 60)

    def _train_orpo(self):
        """ORPO: SFT на chosen + odds-ratio штраф за rejected — один этап,
        без reward-модели и без генерации (в отличие от PPO), без reference-
        копии весов (в отличие от DPO). Совместим с LoRA/GaLore, LISA, AMP."""
        from dataset_formatter import PreferenceDataset, pref_collate_fn
        from orpo import orpo_loss, avg_logp_from_logits

        if getattr(self.config, "packing", False) or getattr(self.config, "group_by_length", False):
            print("[ORPO] Packing и length grouping в ORPO-режиме не поддерживаются — отключаю.")
            self.config.packing = False
            self.config.group_by_length = False

        orpo_file = getattr(self.config, "orpo_file", None)
        if not orpo_file or not os.path.exists(orpo_file):
            raise SystemExit(
                "[ORPO] Нужен файл пар предпочтений (--orpo_file): jsonl со строками "
                '{"messages_chosen": [...], "messages_rejected": [...]} '
                'или {"prompt": ..., "chosen": ..., "rejected": ...}.')
        if self.config.kl_anchor > 0:
            print("[ORPO] KL-якорь несовместим с ORPO-лоссом — отключаю (kl_anchor=0).")
            self.config.kl_anchor = 0
        beta = getattr(self.config, "orpo_beta", 0.1)

        pref_dataset = PreferenceDataset(
            jsonl_path=orpo_file, tokenizer=self.tokenizer,
            max_length=self.config.max_length, max_samples=self.config.max_samples,
            cache_dir=getattr(self.config, "cache_dir", None))
        train_loader = DataLoader(
            pref_dataset, batch_size=self.config.batch_size,
            collate_fn=lambda b: pref_collate_fn(b, pad_token_id=self.tokenizer.pad_token_id))

        optimizer = self._build_optimizer()
        lr_scheduler, warmup_steps = self._build_scheduler(optimizer)
        self._opt, self._sched = optimizer, lr_scheduler
        self._start_listener()

        pending = self._load_training_state()
        start_epoch = 0
        skip_batches = 0
        global_step = 0
        if pending is not None:
            optimizer, lr_scheduler, global_step, pending = self._apply_pending_state(
                optimizer, lr_scheduler, pending)
            self._opt, self._sched = optimizer, lr_scheduler
            if pending is not None:
                start_epoch = pending["epoch"]
                skip_batches = pending["epoch_batches"]
                print(f"[Продолжение] Начинаю с шага {global_step} "
                      f"(эпоха {start_epoch}, пропускаю {skip_batches} пройденных батчей).")

        rs = getattr(self.config, "resume_step", None)
        if isinstance(rs, str) and rs.lower() == "auto":
            rs = self._latest_checkpoint_step()
        if rs:
            rs = int(rs)
            for _ in range(rs):
                lr_scheduler.step()
            global_step = rs
            start_epoch = 0
            skip_batches = rs * self.config.gradient_accumulation_steps
            print(f"[Продолжение] Режим --resume_step: шаг {rs}, "
                  f"пропускаю {skip_batches} батчей.")
        self._global_step = global_step

        print(f"\n=== НАЧАЛО ORPO-ОБУЧЕНИЯ (beta={beta}) ===")
        print(f"Файл предпочтений: {orpo_file}")
        print(f"Безопасный старт: LR={self.config.learning_rate:.0e} | max_grad_norm={self.config.max_grad_norm} | "
              f"grad_accum={self.config.gradient_accumulation_steps} | warmup={warmup_steps} шагов")
        print("=" * 60)

        self.model.train()
        self._maybe_compile()
        self._setup_amp()
        self._init_lisa_and_guards(
            resume_active=pending.get("lisa_active") if pending else None)

        running_loss = running_sft = running_or = 0.0

        for epoch in range(start_epoch, self.config.num_epochs):
            if self.stop_requested:
                break
            print(f"\n--- Эпоха {epoch + 1}/{self.config.num_epochs} (ORPO) ---")
            self._cur_epoch = epoch
            self._cur_epoch_batches = 0
            step = 0
            iterator = iter(train_loader)
            if epoch == start_epoch and skip_batches > 0:
                print(f"[Продолжение] Пропускаю {skip_batches} уже пройденных батчей эпохи...")
                for i in range(skip_batches):
                    try:
                        next(iterator)
                    except StopIteration:
                        break
                    if (i + 1) % 100 == 0 or (i + 1) == skip_batches:
                        print(f"[Продолжение] Пропущено {i + 1}/{skip_batches}...")
            for batch in iterator:
                if self.stop_requested:
                    break
                self._cur_epoch_batches += 1
                self.res_mgr.throttle_step()

                with torch.autocast(self._amp_device, self._amp_dtype,
                                    enabled=self._amp_enabled):
                    out_ch = self.model(input_ids=batch["chosen"]["input_ids"].to(self.device),
                                        attention_mask=batch["chosen"]["attention_mask"].to(self.device),
                                        labels=batch["chosen"]["labels"].to(self.device))
                    out_rj = self.model(input_ids=batch["rejected"]["input_ids"].to(self.device),
                                        attention_mask=batch["rejected"]["attention_mask"].to(self.device),
                                        labels=batch["rejected"]["labels"].to(self.device))
                if out_ch.loss is None or torch.isnan(out_ch.loss):
                    continue

                avg_ch = avg_logp_from_logits(out_ch.logits, batch["chosen"]["labels"])
                avg_rj = avg_logp_from_logits(out_rj.logits, batch["rejected"]["labels"])
                loss, sft_part, or_part = orpo_loss(avg_ch, avg_rj, out_ch.loss, beta=beta)
                del out_ch, out_rj

                loss = loss / self.config.gradient_accumulation_steps
                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()
                running_loss += loss.item() * self.config.gradient_accumulation_steps
                running_sft += sft_part.item()
                running_or += or_part.item()

                step += 1
                if step % self.config.gradient_accumulation_steps == 0:
                    self._optim_step(optimizer, lr_scheduler)
                    global_step += 1
                    self._global_step = global_step

                    if getattr(self.config, "lisa_layers", 0) and self.config.lisa_layers > 0 \
                            and global_step % self.config.lisa_interval == 0:
                        self._rotate_lisa(global_step)

                    if global_step % self.config.logging_steps == 0:
                        stats = self.res_mgr.get_stats()
                        window = (
                            self.config.logging_steps
                            * self.config.gradient_accumulation_steps
                        )
                        print(
                            f"[Шаг {global_step}] "
                            f"Loss: {running_loss / window:.4f} "
                            f"(sft {running_sft / window:.4f} + "
                            f"or {running_or / window:.4f}) | "
                            f"LR: {lr_scheduler.get_last_lr()[0]:.2e} | "
                            f"RAM процесса: {stats.get('ram_used_mb', 0):.0f}MB | "
                            f"Потоков: {stats.get('active_threads')} | "
                            f"Профиль: {self.config.preset}"
                        )
                        running_loss = running_sft = running_or = 0.0

                    if global_step % self.config.save_steps == 0:
                        self.save_model(f"checkpoint-step-{global_step}")
                        self.save_model("interactive_checkpoint")
                        self._save_training_state(
                            reason="на чекпоинте",
                            weights_dir=os.path.join(self.config.output_dir, "interactive_checkpoint"))
                        gc.collect()

        print("\n" + "=" * 60)
        print(" ORPO-обучение завершено или остановлено.")
        self.save_model("final_model")
        self.save_model("interactive_checkpoint")
        self._save_training_state(
            reason="в финале",
            weights_dir=os.path.join(self.config.output_dir, "interactive_checkpoint"))
        print("=" * 60)

    def save_model(self, subfolder: str = ""):
        save_path = os.path.join(self.config.output_dir, subfolder) if subfolder else self.config.output_dir
        os.makedirs(save_path, exist_ok=True)
        print(f"\n[Сохранение] Сохранение обученных LoRA весов в '{save_path}'...")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

        # mmap-модель не знает своего имени — прописываем базу в adapter_config,
        # иначе resume-проверка при продолжении обучения не сработает
        ac_path = os.path.join(save_path, "adapter_config.json")
        if os.path.exists(ac_path):
            try:
                with open(ac_path, "r", encoding="utf-8") as f:
                    ac_data = json.load(f)
                if not ac_data.get("base_model_name_or_path"):
                    ac_data["base_model_name_or_path"] = self.config.model_name_or_path
                    with open(ac_path, "w", encoding="utf-8") as f:
                        json.dump(ac_data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

        # Вшиваем Forge-личность в chat_template ТОЛЬКО для семейства Gemma:
        # у Qwen/Llama свои канонические шаблоны, их затирать нельзя
        if getattr(self, "_model_family", "unknown") == "gemma":
            tc_path = os.path.join(save_path, "tokenizer_config.json")
            if os.path.exists(tc_path):
                try:
                    with open(tc_path, "r", encoding="utf-8") as f:
                        tc_data = json.load(f)
                    tc_data["chat_template"] = GEMMA4_LITTLE_FORGE_TEMPLATE
                    with open(tc_path, "w", encoding="utf-8") as f:
                        json.dump(tc_data, f, ensure_ascii=False, indent=2)
                    print("[Сохранение] Chat-template с личностью Forge вшит в чекпоинт (gemma)!")
                except Exception as e:
                    print(f"[Сохранение] Предупреждение: не удалось пропатчить chat_template: {e}")
        else:
            print(f"[Сохранение] Семейство '{getattr(self, '_model_family', '?')}': chat-template не трогаем (родной шаблон модели сохранен)")

        print(f"[Сохранение] Готово! Веса сохранены.")
