"""Пре-флайт проверка настроек дообучения.

Запускается приложением: JSON настроек на stdin, JSON отчёта на stdout.
Железо и автоплан считает тот же код, что и тренер (hardware.py,
memory_plan.py), поэтому проверка не расходится с реальным запуском.

Формат отчёта:
  {"hardware": {...}, "plan": {...},
   "checks": [{"level": "ok|warn|error|info", "field": "...", "message": "..."}],
   "errors": N, "warnings": M}
"""

import glob
import gzip
import json
import os
import shutil
import sys

CHECKS = []


def add(level, field, message):
    CHECKS.append({"level": level, "field": field, "message": message})


def _open_text(path):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _cfg_value(cfg, key):
    """Значение из config.json, в т.ч. из text_config у мультимодальных."""
    if key in cfg and cfg[key] is not None:
        return cfg[key]
    text = cfg.get("text_config") or {}
    return text.get(key)


def _count_lines(path):
    if str(path).lower().endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            return max(1, sum(1 for _ in f))
    n = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            n += chunk.count(b"\n")
    return max(1, n)


def _first_json_line(path):
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            return json.loads(line)
    return None


def check_paths(s):
    model = s.get("model") or ""
    if not model:
        add("error", "model", "Не указана модель")
    elif not os.path.isdir(model):
        add("error", "model", f"Папки модели нет: {model}")
    else:
        shards = glob.glob(os.path.join(model, "*.safetensors"))
        if not shards:
            add("error", "model", "В папке модели нет *.safetensors")
        else:
            add("ok", "model", f"Веса найдены ({len(shards)} файл(ов))")
        has_tok = any(
            os.path.exists(os.path.join(model, n))
            for n in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model")
        )
        if not has_tok:
            add("warn", "model", "Нет файлов токенизатора — AutoTokenizer может не загрузиться")

    train = s.get("train_file") or ""
    if not train:
        add("error", "train_file", "Не указан датасет")
    elif not os.path.exists(train):
        add("error", "train_file", f"Файла датасета нет: {train}")
    else:
        try:
            row = _first_json_line(train)
            if row is None:
                add("error", "train_file", "Датасет пустой")
            elif not (row.get("messages") or row.get("conversations")):
                add("warn", "train_file", "В первой строке нет поля messages — форматтер может пропустить все строки")
            else:
                lines = _count_lines(train)
                add("ok", "train_file", f"Датасет читается, ~{lines} строк")
                if s.get("max_samples") and int(s["max_samples"]) > lines:
                    add("warn", "max_samples", f"Лимит {s['max_samples']} больше строк в файле ({lines})")
        except Exception as e:
            add("error", "train_file", f"Первая строка не парсится как JSON: {e}")

    val = s.get("val_file") or ""
    if val and not os.path.exists(val):
        add("info", "val_file", "Файла валидации нет — на обучение не влияет (валидация не запускается)")

    identity = s.get("identity_file") or ""
    if identity and not os.path.exists(identity):
        add("info", "identity_file", "Файла личности нет — подмешивание identity отключится")

    replay = s.get("replay_file") or ""
    if replay and not os.path.exists(replay):
        add("warn", "replay_file", f"Файла репетиции нет: {replay} — anti-forget не включится")

    out = s.get("out") or "output_model"
    try:
        os.makedirs(out, exist_ok=True)
        probe = os.path.join(out, ".write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        add("ok", "out", f"Папка сохранения доступна: {out}")
    except Exception as e:
        add("error", "out", f"В папку сохранения нельзя писать ({out}): {e}")

    if s.get("objective") == "orpo":
        f = s.get("orpo_file") or ""
        if not f:
            add("error", "orpo_file", "Для ORPO нужен файл пар chosen/rejected")
        elif not os.path.exists(f):
            add("error", "orpo_file", f"Файла пар нет: {f}")


def check_limits(s, cfg):
    ml = int(s.get("max_length") or 0)
    if ml <= 0:
        add("error", "max_length", "Контекст должен быть больше нуля")
        return
    ctx = _cfg_value(cfg, "max_position_embeddings")
    if isinstance(ctx, int) and ml > ctx:
        add("error", "max_length", f"Контекст {ml} больше, чем поддерживает модель ({ctx})")
    elif isinstance(ctx, int):
        add("ok", "max_length", f"Контекст {ml} укладывается в модель ({ctx})")
    if ml < 64:
        add("warn", "max_length", "Очень короткий контекст — ответы будут обрезаться")
    elif ml > 2048:
        add("warn", "max_length", "Длинный контекст: на CPU шаг будет заметно медленнее")

    epochs = int(s.get("epochs") or 1)
    if epochs < 1:
        add("error", "epochs", "Эпох должно быть минимум 1")
    elif epochs > 5:
        add("warn", "epochs", f"{epochs} эпох — на потоковом датасете это долго; проверь, что так задумано")

    lr = float(s.get("lr") or 0)
    if lr <= 0:
        add("error", "lr", "Learning rate должен быть больше нуля")
    elif lr > 5e-4:
        add("warn", "lr", f"LR {lr:.1e} высоковат для LoRA — возможны скачки лосса")
    elif lr < 5e-7:
        add("warn", "lr", f"LR {lr:.1e} очень маленький — обучение почти не сдвинется")

    r = int(s.get("lora_r") or 0)
    alpha = int(s.get("lora_alpha") or 0)
    if r <= 0:
        add("error", "lora_r", "LoRA r должен быть больше нуля")
    if alpha <= 0:
        add("error", "lora_alpha", "LoRA alpha должен быть больше нуля")
    elif r > 0 and alpha < r:
        add("warn", "lora_alpha", f"alpha ({alpha}) меньше r ({r}) — адаптер учится слишком слабо (обычно alpha = 2r)")


def check_memory(s, hw, plan, cfg):
    ml = int(s.get("max_length") or 512)
    bs = max(1, int(s.get("batch_size") or 1))
    hidden = _cfg_value(cfg, "hidden_size") or _cfg_value(cfg, "n_embd") or 0
    layers = _cfg_value(cfg, "num_hidden_layers") or _cfg_value(cfg, "n_layer") or 0
    vocab = _cfg_value(cfg, "vocab_size") or 0

    ckpt = bool(s.get("grad_checkpointing", True))
    ram_avail = float(hw.get("ram_avail_gb") or 0)
    vram_free = float(hw.get("vram_free_gb") or 0)
    cuda = bool(hw.get("cuda_available"))

    weights = float(plan.get("weights_gb") or 0)
    if weights > 0:
        if plan.get("strategy") == "ram":
            add("ok", "model", f"Веса {weights:.1f} ГБ влезают в RAM")
        else:
            add("info", "model", f"Веса {weights:.1f} ГБ больше свободной RAM — будет {plan.get('strategy')}-режим")

    pm = plan.get("max_length")
    if pm and ml > int(pm) and plan.get("strategy") != "ram":
        add("warn", "max_length",
            f"Автоплан для этого железа советует контекст {pm} "
            f"(стратегия {plan.get('strategy')}), а выставлен {ml}")

    # Активации: скрытые состояния всех слоёв; чекпоинтинг держит ~корень из слоёв.
    if not (hidden and layers):
        add("info", "model", "В config.json нет hidden_size/num_hidden_layers — оценку активаций пропускаю")
    if hidden and layers:
        factor = 4.0 * (1.0 if ckpt else 1.0)
        act_bytes = hidden * layers * ml * bs * 2 * factor  # bf16
        if ckpt:
            act_bytes = act_bytes / max(1.0, layers ** 0.5)
        act_gb = act_bytes / 1e9
        limit = vram_free if cuda else ram_avail
        if limit > 0:
            if act_gb > 0.8 * limit:
                add("error", "max_length",
                    f"Активации ~{act_gb:.1f} ГБ при контексте {ml} и batch {bs} — "
                    f"не влезут в {limit:.1f} ГБ. Уменьши контекст или batch.")
            elif act_gb > 0.5 * limit:
                add("warn", "max_length",
                    f"Активации ~{act_gb:.1f} ГБ — больше половины доступных {limit:.1f} ГБ, "
                    f"риск OOM. Поможет меньший контекст или grad checkpointing.")
            else:
                add("ok", "max_length", f"Оценка активаций ~{act_gb:.1f} ГБ при {limit:.1f} ГБ — запас есть")

    # Логиты: vocab × токены × fp32 и вторая копия для лосса.
    if vocab:
        logits_gb = vocab * ml * bs * 4 / 1e9
        if logits_gb > 2.0:
            add("warn", "max_length",
                f"Логиты ~{logits_gb:.1f} ГБ (словарь {vocab} × {ml} токенов) — "
                f"при OOM уменьши контекст или batch")
        else:
            add("ok", "max_length", f"Логиты ~{logits_gb:.1f} ГБ — нормально")

    if s.get("tune_mode") == "full":
        if s.get("optimizer") in ("adamw", "adamw8bit"):
            add("warn", "optimizer",
                "Full-режим с AdamW хранит состояния всех весов — памяти может не хватить, "
                "лучше galore")
        add("info", "tune_mode", "Full-режим обучает все веса: медленно и требует много RAM")

    if s.get("optimizer") == "galore" and s.get("tune_mode") != "full":
        add("info", "optimizer", "GaLore раскрывается в режиме full — в lora работает вполсилы")


def check_precision(s, hw):
    amp = bool(s.get("amp", True))
    amp_dtype = s.get("amp_dtype") or "auto"
    cuda = bool(hw.get("cuda_available"))

    if amp and not cuda:
        pref = amp_dtype
        if pref in ("auto", "bf16"):
            try:
                import torch
                native = bool(getattr(torch.cpu, "_is_avx512_bf16_supported", lambda: False)())
            except Exception:
                native = False
            if native:
                add("ok", "amp_dtype", "AMP bf16 на CPU: есть AVX512-BF16 — будет быстрее fp32")
            else:
                add("warn", "amp_dtype", "AMP bf16 на CPU без аппаратной поддержки может быть не быстрее fp32")
        elif pref == "fp16":
            add("warn", "amp_dtype", "fp16 на CPU не ускоряет — тренер сам переключится на bf16")

    if s.get("dtype") == "fp16" and not cuda:
        add("warn", "dtype", "fp16-веса на CPU считаются медленно — лучше auto/bf16")

    q = s.get("quant4")
    if q is True or q == "on":
        if not cuda:
            add("error", "quant4", "4-битные веса (NF4) требуют CUDA — на CPU обучение не стартует")
        elif not hw.get("has_bitsandbytes"):
            add("error", "quant4", "quant4=on, но bitsandbytes не установлен")
    elif q in ("auto", None) and not cuda:
        add("info", "quant4", "4-бит пропускается: CUDA нет (это нормально)")


def check_features(s, hw):
    import importlib.util

    if s.get("attn") == "fa2" and importlib.util.find_spec("flash_attn") is None:
        add("warn", "attn", "flash-attn не установлен — вниманию придётся откатиться на sdpa")
    if s.get("liger"):
        ok = importlib.util.find_spec("liger_kernel") and importlib.util.find_spec("triton")
        if not ok or not hw.get("cuda_available"):
            add("warn", "liger", "Liger ядра доступны только на Linux+CUDA+Triton — здесь будут пропущены")
    if s.get("compile"):
        if not (shutil.which("cl") or shutil.which("gcc") or shutil.which("cc")):
            add("warn", "compile", "Нет компилятора C++ — torch.compile не соберётся и будет пропущен")
        if s.get("lisa_layers") and int(s.get("lisa_interval") or 0) < 200:
            add("warn", "compile", "Частая LISA-ротация заставит перекомпилировать модель")

    if s.get("packing"):
        if s.get("objective") == "orpo":
            add("error", "packing", "Packing несовместим с ORPO — выбери что-то одно")
        if int(s.get("batch_size") or 1) <= 1:
            add("info", "packing", "На batch=1 packing не убирает padding, а внимание дороже — обычно медленнее")
    if s.get("group_by_length") and int(s.get("batch_size") or 1) <= 1:
        add("info", "group_by_length", "На batch=1 length grouping почти ничего не даёт (padding и так нулевой)")
    if s.get("group_by_length") and s.get("packing"):
        add("info", "group_by_length", "При packing length grouping отключается автоматически")
    if s.get("kd"):
        if s.get("objective") == "orpo":
            add("error", "kd", "Дистилляция KD несовместима с ORPO — выбери SFT")
        if s.get("packing"):
            add("warn", "kd", "Packing несовместим с soft-targets — будет отключён")
        if float(s.get("kd_weight") or 0) <= 0:
            add("warn", "kd", "kd_weight = 0 — KD-член не будет влиять на лосс")
        add("info", "kd", "Нужен датасет с полем soft (run_distill.py → train_distill.kd.jsonl.gz)")


def check_packing_fit(s):
    """Если включён packing — хватит ли длины диалогов, чтобы что-то упаковать."""
    if not s.get("packing"):
        return
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(s["model"], trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        ml = int(s.get("max_length") or 512)
        lens = []
        with _open_text(s["train_file"]) as f:
            for i, line in enumerate(f):
                if i >= 24 or len(lens) >= 24:
                    break
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                msgs = row.get("messages") or []
                if not msgs:
                    continue
                try:
                    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
                except Exception:
                    continue
                lens.append(min(len(tok(text, add_special_tokens=False)["input_ids"]), ml))
        if lens:
            avg = sum(lens) / len(lens)
            if avg > 0.9 * ml:
                add("warn", "packing",
                    f"Средний диалог ~{avg:.0f} из {ml} токенов — в окно влезает один, "
                    f"padding не экономится. Увеличь контекст или выключи packing")
            else:
                add("ok", "packing", f"Средний диалог ~{avg:.0f} токенов, в окно {ml} влезает ~{ml / max(1.0, avg):.1f}")
    except Exception as e:
        add("info", "packing", f"Не удалось оценить длины диалогов: {str(e)[:80]}")


def main():
    try:
        raw = sys.stdin.read()
        s = json.loads(raw) if raw.strip() else {}
    except Exception as e:
        print(json.dumps({"checks": [{"level": "error", "field": "settings",
                                      "message": f"Не разобрал настройки: {e}"}],
                          "errors": 1, "warnings": 0, "hardware": {}, "plan": {}}))
        return

    report = {"hardware": {}, "plan": {}, "checks": CHECKS, "errors": 0, "warnings": 0}

    try:
        import hardware
        import memory_plan

        hw = hardware.detect_hardware()
        cfg = {}
        try:
            with open(os.path.join(s.get("model") or "", "config.json"), encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            add("warn", "model", "Не прочитал config.json модели — часть оценок по памяти пропущена")

        plan = memory_plan.plan_for_model(s.get("model") or "", hw, user={})
        report["hardware"] = {
            "cpu": hw.get("cpu_count"),
            "ram_total_gb": round(float(hw.get("ram_total_gb") or 0), 1),
            "ram_avail_gb": round(float(hw.get("ram_avail_gb") or 0), 1),
            "cuda": bool(hw.get("cuda_available")),
            "gpu": hw.get("gpu_name"),
            "vram_gb": round(float(hw.get("vram_total_gb") or 0), 1),
            "disk_free_gb": round(float(hw.get("disk_free_gb") or 0), 1),
            "torch": hw.get("torch_version"),
            "python": sys.executable,
        }
        report["plan"] = {
            "strategy": plan.get("strategy"),
            "weights_gb": round(float(plan.get("weights_gb") or 0), 2),
            "max_length": plan.get("max_length"),
            "batch_size": plan.get("batch_size"),
            "grad_accum": plan.get("gradient_accumulation_steps"),
            "optimizer": plan.get("optimizer"),
            "lisa_layers": plan.get("lisa_layers"),
            "lisa_interval": plan.get("lisa_interval"),
            "lora_r": plan.get("lora_r"),
            "lora_alpha": plan.get("lora_alpha"),
            "save_steps": plan.get("save_steps"),
            "amp": plan.get("amp"),
            "amp_dtype": plan.get("amp_dtype"),
            "gradient_checkpointing": plan.get("gradient_checkpointing"),
            "dtype": plan.get("dtype"),
            "quant4": plan.get("quant4"),
        }

        check_paths(s)
        check_limits(s, cfg)
        check_memory(s, hw, plan, cfg)
        check_precision(s, hw)
        check_features(s, hw)
        check_packing_fit(s)
        if s.get("kd"):
            eff_ml = int(s.get("max_length") or plan.get("max_length") or 512)
            if eff_ml < 1024:
                add("warn", "kd",
                    f"KD: контекст {eff_ml} токенов — длинные ответы будут пропущены. "
                    f"Освободи RAM/подними контекст или генерируй короче (--max-answer-tokens).")

        for w in plan.get("warnings", []) or []:
            add("info", "plan", str(w))
    except Exception as e:
        add("error", "env", f"Проверка окружения упала: {e}")

    report["errors"] = sum(1 for c in CHECKS if c["level"] == "error")
    report["warnings"] = sum(1 for c in CHECKS if c["level"] == "warn")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
