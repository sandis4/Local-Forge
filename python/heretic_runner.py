from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _fail(message: str) -> None:
    raise SystemExit(f"[Heretic] {message}")


def _find_heretic_dir() -> Path:
    candidates = []
    env_dir = os.environ.get("HERETIC_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend([PROJECT_ROOT / "heretic-2.1.0", Path.cwd() / "heretic-2.1.0"])
    for candidate in candidates:
        if (candidate / "src" / "heretic" / "main.py").is_file():
            return candidate
    _fail(
        "не нашёл heretic-2.1.0. Положи папку в корень проекта, "
        "поставь HERETIC_DIR или установи heretic-llm."
    )


def _model_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_dir():
        _fail(f"модель '{value}' не найдена. Heretic нужен локальный HF-каталог.")
    if (path / "adapter_config.json").is_file() and not (path / "config.json").is_file():
        _fail(
            f"'{path}' — LoRA-адаптер, а не merged-модель. "
            "Сначала слей LoRA с базой (например output_little_11/merged_final)."
        )
    if not (path / "config.json").is_file():
        _fail(f"в '{path}' нет config.json — это не HF-модель.")
    has_base_weights = bool(list(path.glob("model*.safetensors"))) or bool(
        list(path.glob("pytorch_model*.bin"))
    )
    if (path / "adapter_config.json").is_file() and not has_base_weights:
        _fail(
            f"'{path}' — LoRA-адаптер, а не merged-модель. "
            "Сначала слей LoRA с базой (например output_little_11/merged_final)."
        )
    if not has_base_weights:
        _fail(f"в '{path}' нет весов модели (*.safetensors или pytorch_model*.bin).")
    return path


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _heretic_command(heretic_dir: Path) -> list[str]:
    exe_name = "heretic.exe" if os.name == "nt" else "heretic"
    venv_exe = heretic_dir / ".venv" / ("Scripts" if os.name == "nt" else "bin") / exe_name
    if venv_exe.is_file():
        return [str(venv_exe)]
    uv = shutil.which("uv")
    if uv:
        return [uv, "run", "heretic"]
    _fail(
        "Heretic не установлен. Выполни: cd heretic-2.1.0 && uv sync "
        "(или pip install heretic-llm)."
    )


def run_heretic(config) -> int:
    model = _model_path(config.model_name_or_path)
    heretic_dir = _find_heretic_dir()
    out_dir = Path(config.output_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()
    save_dir = out_dir / "heretic_adapter"
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    cuda = _cuda_available()
    quant = str(getattr(config, "heretic_quant", "auto") or "auto").lower()
    if quant == "auto":
        quant = "none" if cuda else "bnb4"
    if quant not in ("none", "bnb4", "bnb_4bit"):
        _fail(f"неизвестное квантование '{quant}' (auto|none|bnb4).")
    quant = "bnb4" if quant in ("bnb4", "bnb_4bit") else "none"

    if cuda:
        profile_name = "config.default.toml"
    else:
        profile_name = "config.cpu.toml" if quant == "none" else "config.cpu4bit.toml"
    profile = heretic_dir / profile_name
    if not profile.is_file():
        profile = heretic_dir / "config.default.toml"
    if not profile.is_file():
        _fail(f"не нашёл профиль Heretic ({profile_name}).")

    trials = max(1, int(getattr(config, "heretic_trials", 3) or 3))
    startup = max(0, int(getattr(config, "heretic_startup_trials", 1) or 0))
    startup = min(startup, trials)

    cmd = _heretic_command(heretic_dir)
    cmd += [
        "--model", str(model),
        "--n-trials", str(trials),
        "--n-startup-trials", str(startup),
        "--device-map", "auto" if cuda else "cpu",
        "--save-directory", str(save_dir),
        "--model-action", "save",
        "--export-strategy", "ADAPTER",
        "--trial-index", "0",
        "--checkpoint-action", "restart",
        "--max-response-length", "100" if cuda else "32",
        "--batch-size", "0" if cuda else "1",
        "--max-batch-size", "128" if cuda else "1",
    ]
    if quant == "bnb4" and cuda:
        cmd += ["--quantization", "BNB_4BIT"]

    if os.environ.get("HERETIC_DRY_RUN") == "1":
        print("[Heretic] DRY RUN:", " ".join(cmd), flush=True)
        return 0

    env = os.environ.copy()
    env["HERETIC_CONFIG_FILE"] = str(profile)
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["NO_PROXY"] = "localhost,127.0.0.1"
    env["no_proxy"] = "localhost,127.0.0.1"

    print("=" * 60)
    print(" HERETIC — удаление цензуры без обучения (ARA)")
    print(f" Модель: {model}")
    print(f" Устройство: {'cuda' if cuda else 'cpu'} | квантование: {quant}")
    print(f" Trials: {trials} (startup {startup})")
    print(f" Адаптер: {save_dir}")
    print(" Примечание: Heretic использует свои prompt-наборы; --train_file не участвует.")
    print("=" * 60, flush=True)

    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(
        cmd,
        cwd=str(heretic_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=flags,
    )
    try:
        assert process.stdout is not None
        for line in process.stdout:
            print(line.rstrip(), flush=True)
        code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        print("\n[Heretic] Остановлено пользователем.", flush=True)
        return 130

    if code != 0:
        _fail(f"Heretic завершился с кодом {code}.")

    if not (save_dir / "adapter_config.json").is_file():
        print(
            f"[Heretic] Внимание: adapter_config.json не найден в {save_dir}. "
            "Проверь вывод выше.",
            flush=True,
        )
    else:
        print(f"[Heretic] Готово. Адаптер сохранён: {save_dir}", flush=True)
    return 0
