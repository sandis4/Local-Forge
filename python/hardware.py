"""
Автодетект железа для универсального обучения.
Отвечает на вопрос "что у нас есть": CPU, RAM, VRAM (CUDA/без), диск,
поддержка bf16, наличие библиотек квантования. Без сторонних зависимостей.
"""

import multiprocessing
import os
import shutil
import sys


def _safe_import(name):
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "?")
        return True, str(ver)
    except Exception:
        return False, None


def detect_hardware():
    """Возвращает словарь с ресурсами машины. Никогда не падает."""
    info = {}

    # CPU
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["torch_cuda_build"] = getattr(torch.version, "cuda", None)
    except Exception:
        torch = None
        info["torch_version"] = "unknown"

    info["cpu_count"] = multiprocessing.cpu_count()

    # RAM
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["ram_total_gb"] = vm.total / 1e9
        info["ram_avail_gb"] = vm.available / 1e9
        info["ram_percent"] = vm.percent
    except Exception:
        info["ram_total_gb"] = 0.0
        info["ram_avail_gb"] = 0.0
        info["ram_percent"] = 0.0

    # VRAM / CUDA
    info["cuda_available"] = False
    info["gpu_name"] = None
    info["vram_total_gb"] = 0.0
    try:
        if torch is not None and torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1e9
            )
            free_b, total_b = torch.cuda.mem_get_info(0)
            info["vram_free_gb"] = free_b / 1e9
        else:
            info["vram_free_gb"] = 0.0
    except Exception:
        info["vram_free_gb"] = 0.0

    # Диск (для offload / mmap)
    try:
        offload_root = os.path.abspath(os.getcwd())
        usage = shutil.disk_usage(offload_root)
        info["disk_free_gb"] = usage.free / 1e9
    except Exception:
        info["disk_free_gb"] = 0.0

    # Возможности
    try:
        info["bf16_supported"] = (
            torch is not None and torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ) if torch is not None else False
    except Exception:
        info["bf16_supported"] = False

    has_bnb, v_bnb = _safe_import("bitsandbytes")
    has_quanto, v_quanto = _safe_import("quanto")
    has_optimum, v_optimum = _safe_import("optimum")
    info["has_bitsandbytes"] = has_bnb
    info["has_quanto"] = has_quanto
    info["has_optimum"] = has_optimum

    try:
        import transformers, peft, accelerate
        info["transformers_version"] = transformers.__version__
        info["peft_version"] = peft.__version__
        info["accelerate_version"] = accelerate.__version__
    except Exception:
        pass

    info["platform"] = sys.platform
    return info


def describe_hardware(info=None):
    """Человекочитаемая строка для логов."""
    info = info or detect_hardware()
    lines = []
    lines.append(f"CPU потоков: {info.get('cpu_count')}")
    lines.append(
        f"RAM: всего {info.get('ram_total_gb', 0):.1f} ГБ, "
        f"свободно {info.get('ram_avail_gb', 0):.1f} ГБ"
    )
    if info.get("cuda_available"):
        lines.append(
            f"GPU: {info.get('gpu_name')} "
            f"VRAM {info.get('vram_total_gb', 0):.1f} ГБ "
            f"(свободно {info.get('vram_free_gb', 0):.1f} ГБ)"
        )
    else:
        lines.append("GPU CUDA: нет (режим CPU / iGPU) — обучение через RAM+диск")
    lines.append(f"Диск свободно: {info.get('disk_free_gb', 0):.1f} ГБ")
    quant = []
    if info.get("has_bitsandbytes"):
        quant.append("bitsandbytes")
    if info.get("has_quanto"):
        quant.append("quanto")
    lines.append("Квантование: " + (", ".join(quant) if quant else "не установлено (будет mmap bf16/fp32)"))
    return "\n".join("  " + l for l in lines)


if __name__ == "__main__":
    import json
    info = detect_hardware()
    print(describe_hardware(info))
    print(json.dumps(info, indent=2, ensure_ascii=False, default=str))
