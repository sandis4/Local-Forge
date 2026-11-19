"""
Hydro 1.1 — Google Colab launcher.

Один запуск ячейки:
  1. монтирует Google Drive и ищет .gguf (MyDrive + Shared drives);
  2. если файл один — запускает его, иначе предлагает выбрать номер;
  3. поднимает llama-server на GPU (с откатом на CPU);
  4. поднимает OpenAI-совместимый прокси без API-ключа, который всегда
     добавляет системный промпт Hydro и сохраняет системный промпт IDE;
  5. открывает публичный HTTPS-URL через cloudflared.

Внимание: получившийся URL публичный и без ключа — не публикуй его
где попало, пока сессия Colab жива.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Настройки
# ---------------------------------------------------------------------------
HYDRO_SYSTEM = (
    "ты - Hydro, был создан энтузиастом Forge AI, имеешь всего 5.5B "
    "параметров, являешься до обученной версией Gemma 4 E2B."
)
LLAMA_PORT = 8080
PROXY_PORT = 8000
CTX = 8192
WORK = Path("/content/hydro")
WORK.mkdir(parents=True, exist_ok=True)

# Чтобы кривой системный NO_PROXY не ломал скачивание датасетов/бинарей.
for _k in ("NO_PROXY", "no_proxy"):
    os.environ[_k] = "localhost,127.0.0.1"


# ---------------------------------------------------------------------------
# 1. Google Drive и поиск GGUF
# ---------------------------------------------------------------------------
def mount_drive():
    try:
        from google.colab import drive
    except Exception as e:
        raise SystemExit("Запусти это в Google Colab: %s" % e)
    if not os.path.isdir("/content/drive/MyDrive"):
        print("Монтирую Google Drive…")
        drive.mount("/content/drive")


def find_models():
    roots = [Path("/content/drive/MyDrive"), Path("/content/drive/Shareddrives")]
    split_re = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)
    found = {}
    for root in roots:
        if not root.exists():
            continue
        print("Сканирую", root)
        for dp, dns, fns in os.walk(root):
            dns[:] = [d for d in dns if d not in (".Trash", "Trash")]
            for fn in fns:
                low = fn.lower()
                if not low.endswith(".gguf") or "mmproj" in low or "projector" in low:
                    continue
                p = Path(dp) / fn
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                if size < 1024 * 1024:
                    continue
                m = split_re.match(fn)
                if m and int(m.group(2)) != 1:
                    continue  # остальные шарды llama-server подхватит сам
                real = os.path.realpath(str(p))
                found[real] = (p, size)
    models = [{"path": str(p), "size": s, "name": Path(real).name}
              for real, (p, s) in found.items()]
    models.sort(key=lambda x: x["size"], reverse=True)
    return models


def choose_model(models):
    if not models:
        raise SystemExit("На Google Drive не найдено ни одного .gguf.")
    if len(models) == 1:
        m = models[0]
        print("Найден единственный GGUF — запускаю его.")
        print("  %s  (%.2f ГБ)" % (m["name"], m["size"] / 1e9))
        return m
    print("\nНайдено несколько GGUF. Какую модель запустить?\n")
    for i, m in enumerate(models, 1):
        print("  %2d) %-45s %7.2f ГБ" % (i, m["name"], m["size"] / 1e9))
        print("      %s" % m["path"])
    while True:
        raw = input("\nНомер модели (1-%d): " % len(models)).strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(models):
                return models[idx]
        except ValueError:
            pass
        print("Введи число от 1 до %d." % len(models))


# ---------------------------------------------------------------------------
# 2. llama.cpp: готовый CUDA/CPU бандл с GitHub
# ---------------------------------------------------------------------------
RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"
LLAMA_DIR = WORK / "llama"


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "hydro-colab/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def find_asset(releases, pattern):
    rx = re.compile(pattern)
    for rel in releases:
        for a in rel.get("assets", []):
            if rx.match(a["name"]):
                return a
    return None


def download(url, dst, label):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("Скачиваю %s…" % label)
    urllib.request.urlretrieve(url, dst)
    print("  готово: %.1f МБ" % (dst.stat().st_size / 1e6))


def extract_tar(archive):
    with tarfile.open(archive, "r:*") as tf:
        tf.extractall(LLAMA_DIR)
    try:
        archive.unlink()
    except OSError:
        pass


def has_gpu():
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    return subprocess.run([exe], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode == 0


def install_llama(use_cuda):
    print("\nИщу сборку llama.cpp…")
    releases = get_json(RELEASES_API)
    if use_cuda:
        bin_asset = find_asset(releases, r"^llama-b\d+-bin-ubuntu-cuda-12\.8-x64\.tar\.gz$")
        rt_asset = find_asset(releases, r"^cudart-llama-b\d+-bin-ubuntu-cuda-12\.8-x64\.tar\.gz$")
    else:
        bin_asset = find_asset(releases, r"^llama-b\d+-bin-ubuntu-x64\.tar\.gz$")
        rt_asset = None
    if not bin_asset:
        raise RuntimeError("не нашёл готовую сборку llama.cpp; проверь GitHub releases")
    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    bin_tar = LLAMA_DIR / bin_asset["name"]
    download(bin_asset["browser_download_url"], bin_tar, bin_asset["name"])
    extract_tar(bin_tar)
    if rt_asset:
        rt_tar = LLAMA_DIR / rt_asset["name"]
        download(rt_asset["browser_download_url"], rt_tar, rt_asset["name"])
        extract_tar(rt_tar)

    servers = [p for p in LLAMA_DIR.rglob("llama-server") if p.is_file()]
    if not servers:
        raise RuntimeError("llama-server не найден в архиве")
    server = servers[0]
    os.chmod(server, 0o755)
    libdirs = {str(p.parent) for p in LLAMA_DIR.rglob("*.so*")}
    libdirs.add(str(server.parent))
    return server, sorted(libdirs)


# ---------------------------------------------------------------------------
# 3. llama-server
# ---------------------------------------------------------------------------
def wait_http(url, timeout, proc=None, log_path=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            tail = ""
            if log_path and Path(log_path).exists():
                tail = Path(log_path).read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError("процесс завершился. Лог:\n" + tail)
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("не дождался %s за %d c" % (url, timeout))


def start_llama(server, libdirs, model, gpu, ctx=None):
    ctx = int(ctx or CTX)
    env = os.environ.copy()
    if libdirs:
        env["LD_LIBRARY_PATH"] = ":".join(libdirs) + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        str(server), "-m", model, "-c", str(ctx),
        "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
        "--no-webui", "--jinja",
        "-t", str(max(1, os.cpu_count() or 2)),
    ]
    cmd += ["-ngl", "99"] if gpu else ["-ngl", "0"]
    log_path = WORK / "llama-server.log"
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    print("Запускаю llama-server (%s)…" % ("CUDA" if gpu else "CPU"))
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env)
    try:
        wait_http("http://127.0.0.1:%d/health" % LLAMA_PORT, 600, proc, str(log_path))
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        raise
    print("llama-server готов: контекст %d" % ctx)
    return proc, log


# ---------------------------------------------------------------------------
# 4. Прокси: Hydro system prompt + без ключа
# ---------------------------------------------------------------------------
PROXY_CODE = r'''
import os

# Тот же кривой NO_PROXY с [::1] ломает httpx и в Colab.
for _k in ("NO_PROXY", "no_proxy"):
    os.environ[_k] = "localhost,127.0.0.1"

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

HYDRO_SYSTEM = os.environ["HYDRO_SYSTEM"]
MODEL_ID = os.environ["HYDRO_MODEL_ID"]
UPSTREAM = os.environ["HYDRO_UPSTREAM"]

app = FastAPI(title="Hydro 1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"ok": True, "model": MODEL_ID}


@app.get("/v1/models")
@app.get("/models")
async def models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": 0,
            "owned_by": "forge-ai",
        }],
    }


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    systems = [HYDRO_SYSTEM]
    clean = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") == "system":
            text = str(m.get("content") or "").strip()
            if text:
                systems.append(text)
        else:
            clean.append(m)
    body["messages"] = [{"role": "system", "content": "\n\n".join(systems)}] + clean
    body["model"] = MODEL_ID

    if body.get("stream"):
        client = httpx.AsyncClient(timeout=None)
        upstream_req = client.build_request("POST", UPSTREAM, json=body)
        upstream = await client.send(upstream_req, stream=True)
        if upstream.status_code >= 400:
            data = await upstream.aread()
            await upstream.aclose()
            await client.aclose()
            return Response(data, status_code=upstream.status_code,
                            media_type="application/json")
        async def gen():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()
                await client.aclose()
        return StreamingResponse(gen(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(UPSTREAM, json=body)
    return Response(r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type", "application/json"))
'''


def start_proxy(model_id):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "fastapi", "uvicorn", "httpx"], check=True)
    proxy_path = WORK / "hydro_proxy.py"
    proxy_path.write_text(PROXY_CODE, encoding="utf-8")
    env = os.environ.copy()
    env["HYDRO_SYSTEM"] = HYDRO_SYSTEM
    env["HYDRO_MODEL_ID"] = model_id
    env["HYDRO_UPSTREAM"] = "http://127.0.0.1:%d/v1/chat/completions" % LLAMA_PORT
    log_path = WORK / "proxy.log"
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    print("Запускаю прокси с промптом Hydro…")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "hydro_proxy:app",
         "--host", "127.0.0.1", "--port", str(PROXY_PORT), "--log-level", "warning"],
        cwd=str(WORK), env=env, stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
    )
    wait_http("http://127.0.0.1:%d/health" % PROXY_PORT, 120, proc, str(log_path))
    print("Прокси готов (без API-ключа).")
    return proc, log


# ---------------------------------------------------------------------------
# 5. Публичный HTTPS через cloudflared
# ---------------------------------------------------------------------------
CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"


def start_tunnel():
    cf = WORK / "cloudflared"
    if not cf.exists():
        print("Скачиваю cloudflared…")
        urllib.request.urlretrieve(CLOUDFLARED_URL, cf)
        os.chmod(cf, 0o755)
    log_path = WORK / "cloudflared.log"
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    print("Открываю публичный туннель…")
    proc = subprocess.Popen(
        [str(cf), "tunnel", "--url", "http://127.0.0.1:%d" % PROXY_PORT,
         "--no-autoupdate"],
        stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    )
    deadline = time.time() + 120
    while time.time() < deadline:
        if proc.poll() is not None:
            log.flush()
            tail = Path(log_path).read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError("cloudflared завершился:\n" + tail)
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", text)
        if m:
            return proc, log, m.group(0)
        time.sleep(1)
    raise RuntimeError("cloudflared не выдал URL; смотри %s" % log_path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    mount_drive()
    models = find_models()
    selected = choose_model(models)
    model = selected["path"]
    model_id = re.sub(r"-\d{5}-of-\d{5}$", "", Path(model).stem)

    print("\nМодель: %s" % model)
    print("ID для API: %s" % model_id)

    gpu = has_gpu()
    print("GPU:", "да" if gpu else "нет")
    try:
        server, libdirs = install_llama(gpu)
        llama_proc, llama_log = start_llama(server, libdirs, model, gpu)
    except Exception as e:
        if not gpu:
            raise
        print("\nCUDA-запуск не удался (%s). Пробую CPU…" % e)
        server, libdirs = install_llama(False)
        llama_proc, llama_log = start_llama(server, libdirs, model, False)

    proxy_proc, proxy_log = start_proxy(model_id)
    tunnel_proc, tunnel_log, base = start_tunnel()

    print("\n" + "=" * 68)
    print("HYDRO 1.1 ЗАПУЩЕН")
    print("=" * 68)
    print("Base URL : %s/v1" % base)
    print("Model    : %s" % model_id)
    print("API key  : не требуется")
    print("System   : Hydro + системный промпт клиента/IDE")
    print("\nПример:")
    print("curl %s/v1/chat/completions \\" % base)
    print("  -H 'Content-Type: application/json' \\")
    print("  -d '{\"model\":\"%s\",\"messages\":[{\"role\":\"user\",\"content\":\"Привет\"}],\"stream\":true}'"
          % model_id)
    print("\nURL публичный, пока эта ячейка Colab работает. Не публикуй его зря.")
    print("Останови ячейку, чтобы закрыть сервер.\n")

    try:
        while True:
            for name, proc, log_path in (
                ("llama-server", llama_proc, WORK / "llama-server.log"),
                ("proxy", proxy_proc, WORK / "proxy.log"),
                ("cloudflared", tunnel_proc, WORK / "cloudflared.log"),
            ):
                if proc.poll() is not None:
                    tail = Path(log_path).read_text(encoding="utf-8", errors="replace")[-2500:]
                    print("%s завершился. Лог:\n%s" % (name, tail))
                    return
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nОстанавливаю…")
    finally:
        for proc in (tunnel_proc, proxy_proc, llama_proc):
            try:
                proc.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    main()
