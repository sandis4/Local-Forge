"""
Универсальный загрузчик моделей для дообучения на CPU.

Обычная загрузка, если веса влезают в RAM. Если модель больше доступной памяти
(например Qwen3.5-9B = ~18 ГБ bf16 на машине с 16 ГБ RAM) — mmap-загрузка:
скелет модели строится на meta-устройстве, веса подставляются zero-copy
из safetensors (страницы читаются с диска по мере обращения).

Работает с любыми HF-архитектурами (Qwen, Gemma, Llama и т.д.).
"""

import glob
import os

import psutil
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM

_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "F64": torch.float64}
_DTYPE_SIZES = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}

# Части имени модуля, которые не трогаем LoRA-фильтром (мультимодальные башни)
VISION_AUDIO_HINTS = ("visual", "vision", "audio", "multi_modal", "projector", "patch_embed", "conv")


def _shard_paths(model_dir: str):
    return sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))


def build_shard_index(model_dir: str):
    """Карта key -> путь шарда + суммарный размер весов + dtype первого float-тензора."""
    index, total_bytes, first_dtype = {}, 0, torch.float32
    for path in _shard_paths(model_dir):
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                sl = f.get_slice(key)
                shape, dt = sl.get_shape(), sl.get_dtype()
                numel = 1
                for s in shape:
                    numel *= s
                total_bytes += numel * _DTYPE_SIZES.get(str(dt), 4)
                index.setdefault(key, path)
                if str(dt) in _DTYPES and first_dtype == torch.float32:
                    first_dtype = _DTYPES[str(dt)]
    return index, total_bytes, first_dtype


def _resolve_dtype(dtype_str: str, fallback: torch.dtype) -> torch.dtype:
    return {"auto": fallback, "bf16": torch.bfloat16, "float16": torch.float16,
            "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}.get(dtype_str, fallback)


def _set_param(model, full_name: str, tensor):
    parent = model
    parts = full_name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    parent._parameters[parts[-1]] = torch.nn.Parameter(tensor, requires_grad=False)


def _find_rope_base(cfg):
    """rope_theta может лежать на верхнем уровне или во вложенных конфигах (gemma4: text_config)."""
    for obj in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "vision_config", None),
                getattr(cfg, "audio_config", None)):
        base = getattr(obj, "rope_theta", None)
        if base:
            return float(base)
    return 10000.0


def _rope_scaling_factor(cfg):
    rs = getattr(cfg, "rope_scaling", None) or getattr(getattr(cfg, "text_config", None), "rope_scaling", None)
    if isinstance(rs, dict) and rs.get("rope_type") in ("linear", "llama3") and rs.get("factor"):
        return float(rs["factor"])
    return 1.0


def _reconstruct_buffer(module, bname: str, meta_buf: torch.Tensor, cfg, module_name: str = "") -> torch.Tensor:
    """Запасные формулы для буферов, которых нет ни в чекпоинте, ни в пересозданном модуле."""
    shape, dtype = meta_buf.shape, meta_buf.dtype
    if "inv_freq" in bname:
        length = shape[-1]
        dim = length * 2
        base = _find_rope_base(cfg)
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        return (inv / _rope_scaling_factor(cfg)).reshape(shape)
    if bname == "embed_scale":  # масштаб эмбеддингов
        # per-layer эмбеддинги (PLE) масштабируются по hidden_size_per_layer_input, не по dim веса
        tc = getattr(cfg, "text_config", cfg)
        if "per_layer" in module_name:
            v = float(getattr(tc, "hidden_size_per_layer_input", 0)) or float(getattr(tc, "hidden_size", 1))
        else:
            w = module._parameters.get("weight")
            v = float(w.shape[1]) if (w is not None and w.dim() == 2) else float(getattr(tc, "hidden_size", 1))
        return torch.tensor(v ** 0.5, dtype=dtype)
    if "softcap" in bname:
        cap = 50.0
        for obj in (getattr(module, "config", None), getattr(cfg, "audio_config", None),
                    getattr(cfg, "text_config", None), cfg):
            v = getattr(obj, "attention_logit_cap", None)
            if v:
                cap = float(v)
                break
        return torch.full(shape, cap, dtype=dtype)
    if "timescale" in bname:  # относительные позиционные кодировки (аудио)
        num = shape[-1]
        ts = 1.0 * torch.exp(torch.arange(num, dtype=torch.float32) * -__import__("math").log(10000.0) / max(1, num - 1))
        return ts.reshape(shape).to(dtype)
    return torch.zeros(shape, dtype=dtype)


def _rebuild_module_buffers(module, cfg):
    """Пытается пересоздать модуль из его конфига и скопировать живые буферы.
    Возвращает True, если все мета-буферы модуля закрыты."""
    meta_bufs = [bn for bn, b in module._buffers.items() if b is not None and b.device.type == "meta"]
    if not meta_bufs:
        return True
    mod_cfg = getattr(module, "config", None)
    if mod_cfg is None:
        return False
    fresh = None
    for kwargs in ({"device": torch.device("cpu")}, {}):
        try:
            fresh = type(module)(mod_cfg, **kwargs)
            break
        except Exception:
            continue
    if fresh is None:
        return False
    for bn, bb in fresh._buffers.items():
        if bn in meta_bufs and bb is not None and bb.device.type != "meta":
            old = module._buffers[bn]
            if bb.shape == old.shape and bb.is_floating_point() == old.is_floating_point():
                bb = bb.to(old.dtype)
            module._buffers[bn] = bb
    return all(module._buffers[bn].device.type != "meta" for bn in meta_bufs)


def _alias_keys(index):
    """Алиасы имён ключей шарда: чекпоинт может быть мультимодальным
    (model.language_model.*, model.visual.*), а скелет — текстовым (model.*).
    Возвращает словарь алиас -> оригинальный ключ шарда."""
    alias = {}
    for k in index:
        alias.setdefault(k, k)
        if ".language_model." in k:
            alias.setdefault(k.replace(".language_model.", "."), k)
        if k.startswith("language_model."):
            alias.setdefault("model." + k[len("language_model."):], k)
        if k.startswith("model.language_model."):
            alias.setdefault("model." + k[len("model.language_model."):], k)
    return alias


def _resolve_attn(attn: str = None):
    """auto/sdpa -> None (дефолт transformers, на CPU это sdpa);
    fa2 -> flash_attention_2 при наличии пакета, иначе None + предупреждение."""
    if attn in (None, "auto", "sdpa"):
        return None
    if attn == "fa2":
        import importlib
        if importlib.util.find_spec("flash_attn") is None:
            print("[Attn] Пакет flash-attn не установлен — откат на sdpa.")
            return None
        print("[Attn] FlashAttention-2 включён.")
        return "flash_attention_2"
    print(f"[Attn] Неизвестный бэкенд '{attn}' — использую дефолт.")
    return None


def load_model_auto(model_dir: str, dtype_str: str = "auto", gradient_checkpointing: bool = True,
                    force_mmap: bool = None, attn: str = None, quant4=None):
    """
    Возвращает (model, info_dict). info: mmap=bool, weights_gb=float, dtype=str, quant=bool.
    mmap включается автоматически, если веса занимают > 70% доступной RAM.
    quant4 (None=auto): на CUDA с bitsandbytes грузим базу в NF4 — иначе 9B bf16
    (19 ГБ) не влезет в 16 ГБ T4/P100. Возвращает quant=True (веса уже на GPU).
    """
    if not _shard_paths(model_dir):
        raise FileNotFoundError(f"В {model_dir} нет *.safetensors — нечего загружать")

    index, total_bytes, first_dtype = build_shard_index(model_dir)
    avail = psutil.virtual_memory().available
    # 50% доступной RAM: если веса больше — mmap. На 16 ГБ машине это значит,
    # что всё крупнее ~4 ГБ всегда идёт через mmap (безопасно для работы за ПК)
    need_mmap = force_mmap if force_mmap is not None else (total_bytes > 0.50 * avail)
    dtype = _resolve_dtype(dtype_str, first_dtype)
    attn_impl = _resolve_attn(attn)
    info = {"mmap": need_mmap, "weights_gb": total_bytes / 1e9, "dtype": str(dtype),
            "quant": False}

    # QLoRA: 4-битная база для CUDA (авто — когда bf16 не влезет в VRAM)
    import torch as _torch
    use_cuda = _torch.cuda.is_available()
    vram = 0
    if use_cuda:
        try:
            vram = _torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            vram = 0
    want_quant = (quant4 is True or
                  (quant4 in (None, "auto") and use_cuda and vram > 0
                   and total_bytes / 1e9 > 0.60 * vram))
    if want_quant:
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            qconf = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=_torch.bfloat16)
            kw = dict(quantization_config=qconf, device_map="auto",
                      low_cpu_mem_usage=True)
            if attn_impl:
                kw["attn_implementation"] = attn_impl
            model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
            if gradient_checkpointing:
                try:
                    model.gradient_checkpointing_enable()
                except Exception:
                    pass
            info["quant"] = True
            info["mmap"] = False
            print(f"[Loader] QLoRA: база в NF4-4bit на GPU "
                  f"({total_bytes/1e9:.1f} ГБ -> ~{total_bytes/1e9/4:.1f} ГБ VRAM).")
            return model, info
        except Exception as e:
            print(f"[Loader] 4-bit не взлетел ({str(e)[:150]}) — еду обычным путём.")
            if quant4 is True:
                raise RuntimeError(f"--quant4 on требует bitsandbytes+CUDA: {e}")

    if not need_mmap:
        kw = dict(dtype=dtype, low_cpu_mem_usage=True)
        if attn_impl:
            kw["attn_implementation"] = attn_impl
        model = AutoModelForCausalLM.from_pretrained(model_dir, **kw)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model, info

    print(f"[Loader] Веса {total_bytes/1e9:.1f} ГБ > 50% доступной RAM ({avail/1e9:.1f} ГБ) — mmap-загрузка")
    cfg = AutoConfig.from_pretrained(model_dir)
    if attn_impl:
        try:
            cfg.attn_implementation = attn_impl
        except Exception:
            pass
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=first_dtype)

    alias = _alias_keys(index)
    handles = {}

    def get_tensor(key):
        path = index[key]
        if path not in handles:
            handles[path] = safe_open(path, framework="pt", device="cpu")
        return handles[path].get_tensor(key)

    # 1) Подстановка весов (zero-copy) по точным именам и алиасам
    filled = {}
    for name, module in model.named_modules():
        for pname, p in list(module._parameters.items()):
            if p is None:
                continue
            full = f"{name}.{pname}" if name else pname
            key = alias.get(full)
            if key is not None:
                w = get_tensor(key)
                module._parameters[pname] = torch.nn.Parameter(w, requires_grad=False)
                filled[full] = w

    # 2) Догружаем пропуски (ties вроде lm_head.weight, нестандартные ключи)
    for full, p in model.named_parameters(remove_duplicate=False):
        if p is None or p.device.type != "meta":
            continue
        key = alias.get(full)
        if full in filled:
            _set_param(model, full, filled[full])
        elif key is not None:
            w = get_tensor(key)
            _set_param(model, full, w)
            filled[full] = w
        elif full.endswith("lm_head.weight"):
            # Tied-голова: берём тензор эмбеддингов (в любом месте иерархии)
            src_key = next((k for k in list(filled) + list(index) if k.endswith("embed_tokens.weight")), None)
            if src_key is not None:
                w = filled[src_key] if src_key in filled else get_tensor(src_key)
                _set_param(model, full, w)
                filled[full] = w

    # 3) Буферы: persistent из чекпоинта; для non-persistent сначала пересоздаём модуль
    #    целиком (знает свой rope_theta/масштабы), формулы — запасной путь
    for name, module in model.named_modules():
        has_meta_buf = any(b is not None and b.device.type == "meta" for b in module._buffers.values())
        if not has_meta_buf:
            continue
        closed = _rebuild_module_buffers(module, cfg)
        for bname, b in list(module._buffers.items()):
            if b is None or b.device.type != "meta":
                continue
            if closed:
                break
            full = f"{name}.{bname}" if name else bname
            key = alias.get(full)
            if key is not None:
                module._buffers[bname] = get_tensor(key)
            else:
                module._buffers[bname] = _reconstruct_buffer(module, bname, b, cfg, module_name=name)
        if not closed:
            still = [bn for bn, b in module._buffers.items() if b is not None and b.device.type == "meta"]
            if still:
                print(f"[Loader] ВНИМАНИЕ: у {name} не закрылись буферы: {still}")

    # 4) Контроль: meta-тензоров не осталось
    leftovers = [n for n, p in model.named_parameters(remove_duplicate=False) if p is not None and p.device.type == "meta"]
    leftovers += [f"{n}.{bn}" for n, m in model.named_modules() for bn, b in m._buffers.items()
                  if b is not None and b.device.type == "meta"]
    if leftovers:
        raise RuntimeError(f"После mmap-загрузки остались meta-тензоры: {leftovers[:10]}")

    model._mmap_handles = handles  # держим дескрипторы живыми, пока живы mmap-веса
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    gc_collect = __import__("gc").collect()
    print(f"[Loader] mmap-загрузка завершена: {len(filled)} тензоров, {total_bytes/1e9:.1f} ГБ на диске")
    return model, info
