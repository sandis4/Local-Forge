"""Движок чата Local Forge.

Протокол: JSON-строки на stdin, JSON-события на stdout (по одной на строку).

Команды:
  {"cmd":"load","model":"...","adapter":"","threads":8,"ctx":2048}
  {"cmd":"chat","messages":[...],"max_tokens":256,"temp":0.7,"top_p":0.95,
   "think":false,"web":false,"web_engine":"duckduckgo"}
  {"cmd":"agent","task":"...","root":"...","think":false,"web":false}
  {"cmd":"stop"}
  {"cmd":"shutdown"}

События:
  {"ev":"loading","message":"..."}
  {"ev":"ready","backend":"gguf|transformers","info":"..."}
  {"ev":"token","text":"..."}
  {"ev":"reasoning","text":"..."}
  {"ev":"web","engine":"bing","query":"...","found":5}
  {"ev":"tool","name":"write_file","args":{...}}
  {"ev":"tool_result","name":"write_file","ok":true,"text":"..."}
  {"ev":"done","tokens":N,"tok_s":1.23,"backend":"..."}
  {"ev":"error","message":"..."}

Бэкенды:
  * GGUF  — llama-server из llama_bin/: CPU-инференс с mmap, KV-кэш между
    репликами, потоковая выдача. Это быстрый путь.
  * transformers — safetensors + LoRA через model_loader (mmap/offload),
    медленнее, но работает с обученными адаптерами.
"""

import base64
import html
import ipaddress
import json
import os
import queue
import re
import socket
import ssl
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Протокол — UTF-8 JSON: иначе на Windows с cp1252 кириллица ломает вывод.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

CTX_DEFAULT = 2048
SYSTEM_PROMPT = "You are Local Forge, a helpful AI assistant. Answer in the user language."
API_KEY = "1"
API_MODEL = "current-activated"
_EMIT_LOCK = threading.Lock()


def emit(event):
    line = json.dumps(event, ensure_ascii=False) + "\n"
    with _EMIT_LOCK:
        sys.stdout.write(line)
        sys.stdout.flush()


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def find_llama_server():
    for c in (os.path.join("llama_bin", "llama-server.exe"),
              os.path.join("llama_bin", "llama-server")):
        if os.path.exists(c):
            return os.path.abspath(c)
    return None


_PHYSICAL_CORES = None


def physical_cores():
    global _PHYSICAL_CORES
    if _PHYSICAL_CORES is None:
        _PHYSICAL_CORES = os.cpu_count() or 4
        try:
            import psutil
            n = psutil.cpu_count(logical=False)
            if n:
                _PHYSICAL_CORES = int(n)
        except Exception:
            pass
    return _PHYSICAL_CORES


_HELP_CACHE = {}


def help_text(exe):
    if exe not in _HELP_CACHE:
        text = ""
        try:
            out = subprocess.run(
                [exe, "--help"], capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=10,
                stdin=subprocess.DEVNULL,
                creationflags=0x08000000 if os.name == "nt" else 0)
            text = (out.stdout or "") + (out.stderr or "")
        except Exception:
            pass
        _HELP_CACHE[exe] = text
    return _HELP_CACHE[exe]


def supports_flag(exe, *names):
    text = help_text(exe)
    return any(n in text for n in names)


def has_flash_attn(exe):
    """Старые сборки llama.cpp не знают -fa — проверяем по --help один раз."""
    return supports_flag(exe, "--flash-attn", "-fa")


def detect_radmin_ip():
    try:
        import psutil
        for name, addrs in psutil.net_if_addrs().items():
            if "radmin" not in name.lower():
                continue
            for a in addrs:
                if a.family == socket.AF_INET and not a.address.startswith("127."):
                    return a.address
    except Exception:
        pass
    return None


def model_display_name(backend):
    path = (getattr(backend, "model", None)
            or getattr(backend, "model_path", "")
            or "")
    base = os.path.basename(str(path).replace("\\", "/").rstrip("/"))
    if base.lower().endswith(".gguf"):
        base = base[:-5]
    base = re.sub(r"-\d{5}-of-\d{5}$", "", base)
    return base or "model"


class PublicApi:
    """OpenAI-совместимый доступ к уже загруженной модели.

    Ключ фиксированный (1), модель — current-activated. Наружу отдаём только
    имя модели в system-промпте, без личности и настроек Local Forge.
    """

    def __init__(self, get_backend, get_name, lock, port=8000, host="auto"):
        self.get_backend = get_backend
        self.get_name = get_name
        self.lock = lock
        self.port = int(port)
        self.host = host or "auto"
        self.httpd = None
        self.note = ""

    def status(self):
        running = self.httpd is not None
        return {
            "running": running,
            "host": self.host,
            "port": self.port,
            "key": API_KEY,
            "model": API_MODEL,
            "url": ("http://%s:%d/v1" % (self.host, self.port)) if running else "",
            "note": self.note,
        }

    def start(self):
        if self.host in ("", "auto"):
            ip = detect_radmin_ip()
            if ip:
                self.host = ip
                self.note = "Radmin VPN"
            else:
                self.host = "127.0.0.1"
                self.note = "Radmin VPN не найден — API доступен только с этого компьютера"
        try:
            self._bind()
        except OSError as e:
            if self.host != "127.0.0.1":
                failed = self.host
                self.host = "127.0.0.1"
                self.note = ("не удалось занять %s: %s; API только локально"
                             % (failed, e))
                self._bind()
            else:
                raise
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return self.status()

    def _bind(self):
        handler = type("Handler", (ApiHandler,), {})
        self.httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self.httpd.api = self
        self.httpd.daemon_threads = True

    def stop(self):
        if self.httpd is not None:
            try:
                self.httpd.shutdown()
                self.httpd.server_close()
            except Exception:
                pass
            self.httpd = None


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "LocalForgeAPI/1.0"

    def log_message(self, *args):
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-API-Key")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message, kind="invalid_request_error"):
        self._json(code, {"error": {"message": message, "type": kind}})

    def _authorized(self):
        auth = self.headers.get("Authorization") or ""
        if auth == "Bearer " + API_KEY or self.headers.get("X-API-Key") == API_KEY:
            return True
        self._error(401, "Неверный API-ключ. Используй: Bearer " + API_KEY,
                    "authentication_error")
        return False

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/health":
            self._json(200, {"ok": True, "model": API_MODEL})
            return
        if not self._authorized():
            return
        if path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [{
                "id": API_MODEL,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local-forge",
                "root": self.server.api.get_name(),
            }]})
            return
        if path in ("/v1/models/" + API_MODEL, "/models/" + API_MODEL):
            self._json(200, {
                "id": API_MODEL, "object": "model", "created": int(time.time()),
                "owned_by": "local-forge", "root": self.server.api.get_name(),
            })
            return
        self._error(404, "Нет такого пути: " + path, "not_found")

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path not in ("/v1/chat/completions", "/chat/completions"):
            self._error(404, "Нет такого пути: " + path, "not_found")
            return
        if not self._authorized():
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 4_000_000:
                raise ValueError("пустое или слишком большое тело запроса")
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            self._error(400, "Некорректный JSON: %s" % e)
            return

        api = self.server.api
        backend = api.get_backend()
        if backend is None:
            self._error(503, "Модель не загружена", "model_not_loaded")
            return

        name = api.get_name()
        requested = data.get("model") or API_MODEL
        if requested not in (API_MODEL, name):
            self._error(404, "Модель '%s' недоступна. Используй %s" % (requested, API_MODEL),
                        "model_not_found")
            return

        raw_messages = data.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            self._error(400, "Нужен непустой messages")
            return
        user_messages = []
        for m in raw_messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role == "system":
                continue
            if role not in ("user", "assistant", "tool"):
                continue
            user_messages.append({"role": role, "content": m.get("content") or ""})
        if not any(m["role"] == "user" for m in user_messages):
            self._error(400, "Нужно хотя бы одно сообщение user")
            return
        # Единственный system-промпт — название модели, как просили.
        messages = [{"role": "system", "content": name}] + user_messages

        try:
            max_tokens = max(1, min(4096, int(data.get("max_tokens") or 512)))
        except (TypeError, ValueError):
            max_tokens = 512
        try:
            temp = max(0.0, min(2.0, float(data.get("temperature", 0.7))))
        except (TypeError, ValueError):
            temp = 0.7
        try:
            top_p = max(0.01, min(1.0, float(data.get("top_p", 0.95))))
        except (TypeError, ValueError):
            top_p = 0.95
        stream = bool(data.get("stream"))
        created = int(time.time())
        cid = "chatcmpl-%d-%d" % (created, int(time.time() * 1000) % 100000)

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            stop = threading.Event()

            def send(obj):
                try:
                    self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n")
                                     .encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    stop.set()

            def on_token(text):
                if stop.is_set():
                    return
                send({"id": cid, "object": "chat.completion.chunk", "created": created,
                      "model": API_MODEL,
                      "choices": [{"index": 0, "delta": {"content": text},
                                   "finish_reason": None}]})

            send({"id": cid, "object": "chat.completion.chunk", "created": created,
                  "model": API_MODEL,
                  "choices": [{"index": 0, "delta": {"role": "assistant"},
                               "finish_reason": None}]})
            try:
                with api.lock:
                    tokens, _elapsed = backend.chat(
                        messages, max_tokens, temp, top_p, on_token, stop.is_set,
                        think=False)
                send({"id": cid, "object": "chat.completion.chunk", "created": created,
                      "model": API_MODEL,
                      "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                      "usage": {"prompt_tokens": 0, "completion_tokens": int(tokens),
                                "total_tokens": int(tokens)}})
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception as e:
                    print(f"[API] не удалось отправить [DONE]: {e}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                send({"error": {"message": str(e), "type": "server_error"}})
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except Exception as e:
                    print(f"[API] не удалось отправить [DONE] после ошибки: {e}",
                          file=sys.stderr)
            return

        pieces = []

        def on_token(text):
            pieces.append(text)

        try:
            with api.lock:
                tokens, _elapsed = backend.chat(
                    messages, max_tokens, temp, top_p, on_token, lambda: False,
                    think=False)
        except Exception as e:  # noqa: BLE001
            self._error(500, str(e), "server_error")
            return
        self._json(200, {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": API_MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "".join(pieces)},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": int(tokens),
                      "total_tokens": int(tokens)},
        })


AGENT_SYSTEM = (
    "You are Local Forge Agent working inside the user's project folder. "
    "Use the provided tools to inspect and change files. Never ask for confirmation: "
    "do the work. Paths are relative to the project root. Long outputs are fine — "
    "take as long as you need, nobody will cut you off. If web tools are available, "
    "search for current facts, then open the most relevant URLs with web_fetch and "
    "read the actual page text before answering; cite the URLs. When the task is "
    "done, reply with a short summary and no tool calls."
)

AGENT_TOOLS = [
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List files and folders in the project.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "relative path, '.' for root"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": "Read a text file from the project.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "description": "start line, 0-based"},
            "limit": {"type": "integer", "description": "max lines"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "append_file",
        "description": "Append content to the end of a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "replace_in_file",
        "description": "Replace an exact substring in a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "search": {"type": "string"},
            "replace": {"type": "string"}},
            "required": ["path", "search", "replace"]}}},
    {"type": "function", "function": {
        "name": "run_command",
        "description": "Run a shell command in the project root and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}},
            "required": ["command"]}}},
]


def safe_path(root, rel):
    root_abs = os.path.abspath(root)
    full = os.path.abspath(os.path.join(root_abs, rel or "."))
    if full != root_abs and not full.startswith(root_abs + os.sep):
        raise ValueError("путь вне проекта: %s" % rel)
    return full


WEB_TOOL = {"type": "function", "function": {
    "name": "web_search",
    "description": ("Search the web for current information. Returns titles, URLs, snippets "
                    "and readable text of the top pages."),
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string"}},
        "required": ["query"]}}}

WEB_FETCH_TOOL = {"type": "function", "function": {
    "name": "web_fetch",
    "description": ("Open one http(s) URL from search results and return readable page text. "
                    "Use it to read the actual source instead of relying on snippets."),
    "parameters": {"type": "object", "properties": {
        "url": {"type": "string"},
        "offset": {"type": "integer", "description": "start character, for reading a long page in chunks"},
        "max_chars": {"type": "integer", "description": "max characters of page text (default 12000)"}},
        "required": ["url"]}}}

WEB_TOOLS = [WEB_TOOL, WEB_FETCH_TOOL]

WEB_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
          "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")


def _http_get(url):
    req = urllib.request.Request(url, headers={
        "User-Agent": WEB_UA,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf-8", "replace")


def _plain(markup):
    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", markup,
                  flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _normalize_url(url):
    try:
        p = urllib.parse.urlsplit(url.strip())
    except Exception:
        return ""
    if p.scheme not in ("http", "https") or not p.hostname:
        return ""
    host = p.hostname
    try:
        host = host.encode("idna").decode("ascii")
    except Exception:
        pass
    host = host.lower()
    if ":" in host and not host.startswith("["):
        host = "[" + host + "]"
    netloc = host + (":%d" % p.port if p.port else "")
    path = urllib.parse.quote(p.path or "/", safe="/%:@!$&'()*+,;=-._~")
    query = urllib.parse.quote(p.query, safe="=&?/:;+,%@!$'()*-._~")
    return urllib.parse.urlunsplit((p.scheme, netloc, path, query, ""))


def _public_url_ok(url):
    """Не даём web_fetch ходить на localhost/приватные адреса."""
    url = _normalize_url(url)
    if not url:
        return False
    p = urllib.parse.urlsplit(url)
    host = p.hostname.strip("[]").split("%")[0].lower()
    if host in ("localhost", "localhost.localdomain"):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except Exception:
        return False
    for info in infos:
        addr = str(info[4][0]).split("%")[0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return False
    return True


def _readable_page(markup):
    title = ""
    tm = re.search(r"<title[^>]*>(.*?)</title>", markup, re.S | re.I)
    if tm:
        title = _plain(tm.group(1))[:300]
    body = markup
    for pat in (r"<article\b.*?</article>", r"<main\b.*?</main>"):
        m = re.search(pat, markup, re.S | re.I)
        if m:
            body = m.group(0)
            break
    body = re.sub(r"<(script|style|noscript|svg|nav|footer|header|aside|form)\b"
                  r".*?</\1>", " ", body, flags=re.S | re.I)
    text = _plain(body)
    if len(text) < 300:
        text = _plain(markup)
    return title, text


def fetch_page(url, max_chars=12000, offset=0):
    """Открывает страницу и возвращает (ok, читаемый текст)."""
    url = _normalize_url(url)
    if not _public_url_ok(url):
        return False, "адрес не разрешён (только публичный http/https)"
    req = urllib.request.Request(url, headers={
        "User-Agent": WEB_UA,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
    })
    insecure = False
    try:
        try:
            resp = urllib.request.urlopen(req, timeout=20)
        except Exception as e:
            # На Windows у Python часто нет полного CA-бандла. Для чтения
            # публичной страницы повторяем без проверки и помечаем это в тексте.
            if "CERTIFICATE_VERIFY_FAILED" not in str(e):
                raise
            resp = urllib.request.urlopen(
                req, timeout=20, context=ssl._create_unverified_context())
            insecure = True
        with resp as r:
            final = _normalize_url(r.geturl())
            if not _public_url_ok(final):
                return False, "редирект на запрещённый адрес"
            ctype = (r.headers.get_content_type() or "").lower()
            if ctype not in ("text/html", "application/xhtml+xml", "text/plain"):
                return False, "не текстовая страница (%s)" % (ctype or "?")
            raw = r.read(2_000_001)
            if len(raw) > 2_000_000:
                raw = raw[:2_000_000]
            charset = r.headers.get_content_charset() or "utf-8"
            markup = raw.decode(charset, "replace")
    except Exception as e:  # noqa: BLE001
        return False, "не открылось: %s" % e
    title, text = _readable_page(markup)
    if not text:
        return False, "на странице не нашлось текста"
    try:
        limit = max(500, min(30000, int(max_chars)))
    except (TypeError, ValueError):
        limit = 12000
    try:
        start = max(0, min(len(text), int(offset or 0)))
    except (TypeError, ValueError):
        start = 0
    out = []
    if insecure:
        out.append("[сертификат не проверен]")
    if title:
        out.append("Title: " + title)
    out.append("URL: " + final)
    out.append("")
    out.append(text[start:start + limit])
    rest = len(text) - (start + limit)
    if rest > 0:
        out.append("\n[... ещё %d символов; для продолжения передай offset=%d]" % (rest, start + limit))
    return True, "\n".join(out)


def _bing_url(href):
    """Bing отдаёт редирект bing.com/ck/a?..&u=a1<base64url>."""
    m = re.search(r"[?&]u=a1([^&]+)", href)
    if not m:
        return href
    raw = m.group(1) + "=" * (-len(m.group(1)) % 4)
    try:
        return base64.urlsafe_b64decode(raw).decode("utf-8", "replace")
    except Exception:
        return href


def search_web(query, engine="duckduckgo", limit=5):
    """Бесплатный HTML-поиск без API-ключей. Возвращает (results, error)."""
    q = urllib.parse.quote_plus(query)
    results = []
    note = ""
    try:
        if engine == "bing":
            page = _http_get("https://www.bing.com/search?q=" + q + "&setlang=ru")
            for block in re.findall(r'<li class="b_algo".*?</li>', page, re.S):
                m = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                              block, re.S)
                if not m:
                    continue
                sn = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
                results.append((_plain(m.group(2)), _bing_url(html.unescape(m.group(1))),
                                _plain(sn.group(1)) if sn else ""))
                if len(results) >= limit:
                    break
        elif engine == "google":
            page = _http_get("https://www.google.com/search?q=" + q + "&num=10&hl=ru")
            for m in re.finditer(r'<a href="/url\?q=([^&"]+)[^"]*"[^>]*>(.*?)</a>',
                                 page, re.S):
                url = urllib.parse.unquote(html.unescape(m.group(1)))
                title = _plain(m.group(2))
                if title and url.startswith("http") and "google." not in url:
                    results.append((title, url, ""))
                if len(results) >= limit:
                    break
            if not results:
                for m in re.finditer(
                        r'<a[^>]+href="(https?://[^"]+)"[^>]*>.*?<h3[^>]*>(.*?)</h3>',
                        page, re.S):
                    url = html.unescape(m.group(1))
                    title = _plain(m.group(2))
                    if title and "google." not in url:
                        results.append((title, url, ""))
                    if len(results) >= limit:
                        break
        elif engine == "yandex":
            page = _http_get("https://yandex.ru/search/?text=" + q)
            for m in re.finditer(
                    r'<a[^>]+class="[^"]*(?:OrganicTitle-Link|Link_theme_outer)[^"]*"'
                    r'[^>]+href="([^"]+)"[^>]*>(.*?)</a>', page, re.S):
                url = html.unescape(m.group(1))
                title = _plain(m.group(2))
                if title and url.startswith("http"):
                    results.append((title, url, ""))
                if len(results) >= limit:
                    break
        else:
            page = _http_get("https://html.duckduckgo.com/html/?q=" + q)
            blocks = re.findall(
                r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>'
                r'(.*?)(?=<a[^>]+class="result__a"|$)', page, re.S)
            for href, title, rest in blocks:
                href = html.unescape(href)
                m = re.search(r"[?&]uddg=([^&]+)", href)
                if m:
                    href = urllib.parse.unquote(m.group(1))
                sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', rest, re.S)
                results.append((_plain(title), href,
                                _plain(sn.group(1)) if sn else ""))
                if len(results) >= limit:
                    break
    except Exception as e:  # noqa: BLE001
        note = "ошибка поиска (%s): %s" % (engine, e)
    if not results and engine != "bing":
        alt, _ = search_web(query, "bing", limit)
        if alt:
            return alt, note or ("%s не отдал результаты — показан Bing" % engine)
    if not results and engine not in ("duckduckgo", "bing"):
        alt, _ = search_web(query, "duckduckgo", limit)
        if alt:
            return alt, note or ("%s не отдал результаты — показан DuckDuckGo" % engine)
    if not results:
        return [], note or ("поиск (%s) не дал результатов — возможно, движок "
                            "требует капчу; выбери DDG или Bing" % engine)
    return results, note


def search_results_text(query, engine, fetch_pages=True, max_pages=2, page_chars=3000):
    results, err = search_web(query, engine)
    if not results:
        return False, err
    lines = ["Результаты (%s) по запросу «%s»:" % (engine, query)]
    for i, (title, url, snippet) in enumerate(results, 1):
        lines.append("%d. %s" % (i, title))
        lines.append("   " + url)
        if snippet:
            lines.append("   " + snippet[:500])
        if fetch_pages and i <= max_pages:
            ok, text = fetch_page(url, page_chars)
            if ok:
                lines.append("   [текст страницы]")
                lines.append(text[:page_chars])
            else:
                lines.append("   [страница не открылась: %s]" % text[:160])
    if fetch_pages and len(results) > max_pages:
        lines.append("Остальные ссылки можно открыть через web_fetch.")
    return True, chr(10).join(lines)


def web_context(query, engine, results, fetch_pages=False, max_pages=2,
                page_chars=1600):
    lines = ["Результаты веб-поиска (%s) по запросу: %s" % (engine, query)]
    for i, (title, url, snippet) in enumerate(results, 1):
        lines.append("%d. %s" % (i, title))
        lines.append("   " + url)
        if snippet:
            lines.append("   " + snippet[:400])
        if fetch_pages and i <= max_pages:
            ok, text = fetch_page(url, page_chars)
            if ok:
                lines.append("   [текст страницы]")
                lines.append(text[:page_chars])
            else:
                lines.append("   [страница не открылась: %s]" % text[:200])
    lines.append("Сниппеты — только подсказка; ниже дан текст открытых страниц. "
                 "Опирайся на него, указывай ссылки; если данные расходятся — скажи об этом.")
    return chr(10).join(lines)


def execute_tool(name, args, root, web_engine="duckduckgo"):
    """Выполняет действие агента. Возвращает (ok, текст результата)."""
    try:
        if name == "web_search":
            return search_results_text(args.get("query", ""), web_engine)
        if name == "web_fetch":
            return fetch_page(args.get("url", ""), args.get("max_chars") or 12000,
                              args.get("offset") or 0)
        if name == "list_dir":
            full = safe_path(root, args.get("path", "."))
            items = []
            for e in sorted(os.listdir(full)):
                p = os.path.join(full, e)
                if os.path.isdir(p):
                    items.append(e + "/")
                else:
                    items.append("%s  (%d байт)" % (e, os.path.getsize(p)))
            return True, chr(10).join(items) or "(пусто)"
        if name == "read_file":
            full = safe_path(root, args.get("path", ""))
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            off = int(args.get("offset") or 0)
            lim = int(args.get("limit") or 400)
            chunk = "".join(lines[off:off + lim])
            return True, chunk[:20000] or "(пусто)"
        if name == "write_file":
            full = safe_path(root, args.get("path", ""))
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            data = args.get("content", "")
            with open(full, "w", encoding="utf-8", newline="") as f:
                f.write(data)
            return True, "записано %d символов в %s" % (len(data), args.get("path"))
        if name == "append_file":
            full = safe_path(root, args.get("path", ""))
            os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
            data = args.get("content", "")
            with open(full, "a", encoding="utf-8", newline="") as f:
                f.write(data)
            return True, "дописано %d символов" % len(data)
        if name == "replace_in_file":
            full = safe_path(root, args.get("path", ""))
            with open(full, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            search = args.get("search", "")
            n = text.count(search) if search else 0
            text = text.replace(search, args.get("replace", ""))
            with open(full, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            return True, "заменено вхождений: %d" % n
        if name == "run_command":
            import subprocess
            proc = subprocess.run(
                args.get("command", ""), shell=True, cwd=root,
                capture_output=True, text=True, timeout=900,
                creationflags=(0x08000000 if os.name == "nt" else 0),
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode == 0, ("код %d" % proc.returncode) + chr(10) + out[:20000]
        return False, "неизвестный инструмент: %s" % name
    except Exception as e:  # noqa: BLE001
        return False, "ошибка: %s" % e


class _ThinkTagParser:
    """Режет поток на reasoning внутри <think>...</think> и финальный ответ."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, on_text, on_reason):
        self.on_text = on_text
        self.on_reason = on_reason or on_text
        self.in_reason = False
        self.buf = ""

    @staticmethod
    def _partial_keep(buf, tag):
        for k in range(min(len(tag) - 1, len(buf)), 0, -1):
            if buf.endswith(tag[:k]):
                return k
        return 0

    def _drain(self, final):
        while True:
            tag = self.CLOSE if self.in_reason else self.OPEN
            idx = self.buf.find(tag)
            if idx < 0:
                keep = 0 if final else self._partial_keep(self.buf, tag)
                if final:
                    (self.on_reason if self.in_reason else self.on_text)(self.buf)
                    self.buf = ""
                elif keep:
                    if len(self.buf) > keep:
                        (self.on_reason if self.in_reason else self.on_text)(self.buf[:-keep])
                        self.buf = self.buf[-keep:]
                elif self.buf:
                    (self.on_reason if self.in_reason else self.on_text)(self.buf)
                    self.buf = ""
                return
            if idx:
                (self.on_reason if self.in_reason else self.on_text)(self.buf[:idx])
            self.buf = self.buf[idx + len(tag):]
            self.in_reason = not self.in_reason
            if not self.buf:
                return

    def feed(self, text):
        self.buf += text
        self._drain(False)

    def finish(self):
        self._drain(True)


class _ServerExit(RuntimeError):
    pass


class GgufBackend:
    def __init__(self, model, threads, ctx, kv_type="f16"):
        self.model = model
        self.threads = max(1, min(int(threads), physical_cores()))
        self.ctx = ctx
        self.kv_type = kv_type
        self.proc = None
        self.port = None
        self.native_reasoning = False
        self.tag_reasoning = False
        self.log = os.path.join(os.environ.get("TEMP", "."), "local_forge_llama.log")

    def _detect_reasoning(self):
        try:
            with urllib.request.urlopen(
                    "http://127.0.0.1:%d/props" % self.port, timeout=5) as r:
                props = json.loads(r.read().decode("utf-8", "replace"))
            tpl = (props.get("chat_template") or "").lower()
            self.native_reasoning = any(x in tpl for x in
                                        ("enable_thinking", "reasoning", "<think>", "thinking"))
        except Exception:
            self.native_reasoning = False

    def _prepare_messages(self, messages, think):
        self.tag_reasoning = bool(think and not self.native_reasoning)
        if not self.tag_reasoning:
            return messages
        note = ("Think through the problem. Put the whole reasoning between <think> and "
                "</think>, then write the final answer after </think>. Use the tags exactly "
                "once and do not repeat them in the final answer.")
        out = [dict(m) for m in messages]
        for m in out:
            if m.get("role") == "system":
                m["content"] = (m.get("content") or "") + "\n\n" + note
                return out
        return [{"role": "system", "content": note}] + out

    def _log_tail(self, limit=4000):
        try:
            with open(self.log, encoding="utf-8", errors="replace") as f:
                return f.read()[-limit:]
        except Exception:
            return ""

    @staticmethod
    def _is_memory_error(text):
        low = text.lower()
        return any(s in low for s in (
            "out of memory", "failed to allocate", "unable to fit",
            "not enough memory", "failed to fit",
        ))

    def _explain_exit(self, text):
        low = text.lower()
        hints = []
        if ("check_tensor_dims" in low or "missing tensor" in low
                or ("tensor" in low and "not found" in low)):
            hints.append("GGUF неполный или несовместимый: в файле нет нужных тензоров. "
                         "Перекачай модель целиком, включая все шарды.")
        if self._is_memory_error(text):
            hints.append("Не хватило RAM на KV-кэш или буферы. Веса грузятся через mmap; "
                         "закрой приложения, уменьши контекст или возьми меньший квант.")
        if "unknown model architecture" in low or "unsupported model architecture" in low:
            hints.append("Сборка llama.cpp не знает эту архитектуру — обнови llama-bin.")
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()][-3:]
        tail = "\n".join(ln[-220:] for ln in lines)
        msg = "llama-server завершился (код %s)." % (
            self.proc.returncode if self.proc is not None else "?")
        if hints:
            msg += " " + " ".join(hints)
        if tail:
            msg += "\nПоследние строки лога:\n" + tail
        msg += "\nПолный лог: " + self.log
        return msg

    def start(self):
        exe = find_llama_server()
        if not exe:
            raise RuntimeError("не найден llama_bin/llama-server (или llama-server.exe) — GGUF-путь недоступен")
        try:
            size = os.path.getsize(self.model)
        except OSError:
            size = 0
        try:
            import psutil
            avail = psutil.virtual_memory().available
        except Exception:
            avail = 0
        if size and avail and size > avail:
            emit({"ev": "loading", "message":
                  "Модель %.1f ГБ больше свободной RAM %.1f ГБ — гружу через mmap, будет медленнее."
                  % (size / 1e9, avail / 1e9)})
        flash = has_flash_attn(exe)
        low_mem = False
        try:
            info = self._run(exe, flash, low_mem)
            self._detect_reasoning()
            return info
        except _ServerExit:
            text = self._log_tail()
            if self._is_memory_error(text):
                if self.ctx > 512:
                    self.ctx = max(512, self.ctx // 2)
                emit({"ev": "loading", "message":
                      "Не хватило RAM — пробую контекст %d и меньший batch..." % self.ctx})
                low_mem = True
                try:
                    info = self._run(exe, flash, low_mem)
                    self._detect_reasoning()
                    return info
                except _ServerExit:
                    text = self._log_tail()
            if flash:
                emit({"ev": "loading", "message":
                      "llama-server: flash attention не подошёл, запуск без -fa"})
                try:
                    info = self._run(exe, False, low_mem)
                    self._detect_reasoning()
                    return info
                except _ServerExit:
                    text = self._log_tail()
            raise RuntimeError(self._explain_exit(text))

    def _command(self, exe, flash_attn, port, low_mem=False):
        cmd = [exe, "-m", self.model, "-c", str(self.ctx), "-t", str(self.threads),
               "--host", "127.0.0.1", "--port", str(port), "--no-webui"]
        if supports_flag(exe, "--load-mode"):
            # Веса — через mmap: модель может быть больше свободной RAM,
            # страницы подтягиваются с диска по мере надобности.
            cmd += ["--load-mode", "mmap"]
        elif supports_flag(exe, "--mmap"):
            cmd.append("--mmap")
        if supports_flag(exe, "--fit"):
            # --fit может отказаться грузить модель, которая не влезает в RAM;
            # mmap умеет работать и так. Контекст уменьшим сами при ошибке.
            cmd += ["--fit", "off"]
        if flash_attn:
            # CPU flash attention: меньше KV-памяти и быстрее длинный контекст.
            cmd.append("-fa")
        if low_mem:
            cmd += ["-b", "512", "-ub", "128"]
        if self.kv_type and self.kv_type != "f16":
            # Сжатый KV-кэш экономит RAM на длинном контексте (как в Unsloth).
            cmd += ["-ctk", self.kv_type, "-ctv", self.kv_type]
        return cmd

    def _run(self, exe, flash_attn, low_mem=False):
        self.port = free_port()
        cmd = self._command(exe, flash_attn, self.port, low_mem)
        emit({"ev": "loading", "message": "llama-server: " + os.path.basename(self.model)})
        log = open(self.log, "w", encoding="utf-8", errors="replace")
        no_window = 0x08000000 if os.name == "nt" else 0
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL,
                                     creationflags=no_window)
        deadline = time.time() + 300
        url = "http://127.0.0.1:%d/health" % self.port
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise _ServerExit("llama-server завершился (код %s); лог: %s"
                                  % (self.proc.returncode, self.log))
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return "llama.cpp, контекст %d, потоков %d" % (self.ctx, self.threads)
            except Exception:
                pass
            time.sleep(1.0)
        if self.proc.poll() is None:
            self.proc.kill()
        raise RuntimeError("llama-server не поднялся за 5 минут; лог: %s" % self.log)

    def chat(self, messages, max_tokens, temp, top_p, on_token, should_stop,
             think=False, on_reasoning=None):
        messages = self._prepare_messages(messages, think)
        parser = _ThinkTagParser(on_token, on_reasoning) if self.tag_reasoning else None
        body_obj = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temp),
            "top_p": float(top_p),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if think:
            # llama.cpp: включает reasoning у моделей с шаблоном (Qwen3 и др.).
            body_obj["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            # Не оставляем решение шаблону: у thinking-моделей он может быть
            # включён по умолчанию и съесть весь max_tokens до ответа.
            body_obj["chat_template_kwargs"] = {"enable_thinking": False}
        body = json.dumps(body_obj).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % self.port,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        tokens = 0
        usage_tokens = None
        started = time.time()
        def emit_text(text):
            if parser is not None:
                parser.feed(text)
            else:
                on_token(text)
        try:
            # Без таймаута: долгий ответ и «think» не обрываются.
            with urllib.request.urlopen(req, timeout=None) as resp:
                for raw in resp:
                    if should_stop():
                        break
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except Exception:
                        continue
                    usage = chunk.get("usage") or {}
                    if usage.get("completion_tokens"):
                        usage_tokens = int(usage["completion_tokens"])
                    choices = chunk.get("choices") or []
                    if choices:
                        delta = choices[0].get("delta") or {}
                        reason = delta.get("reasoning_content") or delta.get("reasoning")
                        if reason and on_reasoning:
                            on_reasoning(reason)
                        text = delta.get("content")
                        if text:
                            tokens += 1
                            emit_text(text)
        finally:
            if parser is not None:
                parser.finish()
            elapsed = max(1e-6, time.time() - started)
        return (usage_tokens or tokens), elapsed

    def _request(self, messages, max_tokens, temp, tools=None, on_event=None,
                 should_stop=None, think=False):
        """Один запрос к серверу. Без таймаута: длинная запись файла не рвётся."""
        body = {
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temp),
            "stream": False,
        }
        if think:
            body["chat_template_kwargs"] = {"enable_thinking": True}
        else:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        req = urllib.request.Request(
            "http://127.0.0.1:%d/v1/chat/completions" % self.port,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        result = {}

        def worker():
            try:
                with urllib.request.urlopen(req, timeout=None) as r:
                    result["v"] = json.loads(r.read().decode("utf-8", "replace"))
            except Exception as e:  # noqa: BLE001
                result["err"] = str(e)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while thread.is_alive():
            if should_stop and should_stop():
                return None
            if on_event:
                on_event({"ev": "heartbeat"})
            thread.join(5)
        if "err" in result:
            raise RuntimeError(result["err"])
        return result.get("v")

    def run_agent(self, task, root, max_steps, max_tokens, temp, on_event,
                  should_stop, think=False, web=False, web_engine="duckduckgo"):
        system = AGENT_SYSTEM
        if web:
            system += " Web search is enabled (engine: %s)." % web_engine
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": "Project root: %s%sTask: %s"
             % (root, chr(10), task)},
        ]
        tools = list(AGENT_TOOLS) + (WEB_TOOLS if web else [])
        steps = 0
        for step in range(max_steps):
            if should_stop():
                break
            on_event({"ev": "thinking", "step": step + 1})
            resp = self._request(messages, max_tokens, temp, tools=tools,
                                 on_event=on_event, should_stop=should_stop,
                                 think=think)
            if resp is None:
                on_event({"ev": "assistant", "text": "Остановлено."})
                return steps
            msg = (resp.get("choices") or [{}])[0].get("message") or {}
            calls = msg.get("tool_calls") or []
            if not calls:
                on_event({"ev": "assistant", "text": msg.get("content") or ""})
                return steps
            messages.append({
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": calls,
            })
            for call in calls:
                fn = call.get("function") or {}
                name = fn.get("name") or ""
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                on_event({"ev": "tool", "name": name, "args": args})
                ok, text = execute_tool(name, args, root, web_engine)
                on_event({"ev": "tool_result", "name": name, "ok": ok, "text": text[:4000]})
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id") or name,
                    "content": text[:20000],
                })
                steps += 1
                if should_stop():
                    return steps
        on_event({"ev": "assistant", "text": "Достигнут лимит шагов агента."})
        return steps

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except Exception:
                self.proc.kill()


class TransformersBackend:
    def __init__(self, model, adapter, threads):
        self.model_path = model
        self.adapter = adapter
        self.threads = max(1, min(int(threads), physical_cores()))
        self.model = None
        self.tokenizer = None

    def start(self):
        import torch
        from transformers import AutoTokenizer
        from model_loader import load_model_auto

        torch.set_num_threads(max(1, self.threads))
        emit({"ev": "loading", "message": "Загрузка " + self.model_path + " (mmap/offload)…"})
        tok_src = self.model_path
        if self.adapter and os.path.exists(os.path.join(self.adapter, "tokenizer_config.json")):
            tok_src = self.adapter
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model, info = load_model_auto(self.model_path, gradient_checkpointing=False)
        model.eval()
        if self.adapter and os.path.exists(os.path.join(self.adapter, "adapter_config.json")):
            from peft import PeftModel
            emit({"ev": "loading", "message": "Подключаю LoRA-адаптер…"})
            model = PeftModel.from_pretrained(model, self.adapter)
            try:
                model = model.merge_and_unload()
            except Exception:
                pass
        self.model = model
        where = "mmap" if info.get("mmap") else "RAM"
        return "transformers, веса в " + where

    def chat(self, messages, max_tokens, temp, top_p, on_token, should_stop,
             think=False, on_reasoning=None):
        import torch
        from transformers import (StoppingCriteria, StoppingCriteriaList,
                                  TextIteratorStreamer)

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **(dict(enable_thinking=bool(think))))
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)

        class StopWhen(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return should_stop()

        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=int(max_tokens),
            do_sample=float(temp) > 0,
            temperature=max(1e-4, float(temp)),
            top_p=max(1e-4, float(top_p)),
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList([StopWhen()]),
        )
        holder = {}

        def run():
            try:
                with torch.inference_mode():
                    self.model.generate(**gen_kwargs)
            except Exception as e:  # noqa: BLE001
                holder["error"] = str(e)
            finally:
                try:
                    streamer.end()
                except Exception:
                    pass

        thread = threading.Thread(target=run, daemon=True)
        started = time.time()
        thread.start()
        pieces = 0
        try:
            for piece in streamer:
                if should_stop():
                    break
                if piece:
                    pieces += 1
                    on_token(piece)
        finally:
            elapsed = max(1e-6, time.time() - started)
        if holder.get("error"):
            raise RuntimeError(holder["error"])
        return pieces, elapsed

    def close(self):
        self.model = None


def main():
    commands = queue.Queue()
    stop_flag = threading.Event()

    def reader():
        for line in sys.stdin:
            line = line.strip()
            if line:
                commands.put(line)

    threading.Thread(target=reader, daemon=True).start()

    backend = None
    backend_kind = None
    backend_lock = threading.Lock()
    api = None
    api_enabled = False
    api_port = 8000
    api_host = "auto"

    def emit_api(running=False, url="", note="", error=None, **extra):
        payload = {"running": running, "key": API_KEY, "model": API_MODEL,
                   "url": url or "", "note": note}
        payload.update(extra)
        if error is not None:
            payload["error"] = error
        emit({"ev": "api", **payload})

    def stop_api(announce=True):
        nonlocal api
        if api is not None:
            api.stop()
            api = None
        if announce:
            emit_api(False, note="выключен")

    def start_api(enabled, port, host="auto"):
        nonlocal api
        stop_api(announce=False)
        if not enabled:
            emit_api(False, note="выключен")
            return
        if backend is None:
            emit_api(False, note="Сначала загрузи модель — API поднимется после загрузки")
            return
        try:
            port = max(1024, min(65535, int(port or 8000)))
        except (TypeError, ValueError):
            port = 8000
        server = PublicApi(lambda: backend, lambda: model_display_name(backend),
                           backend_lock, port, host)
        try:
            st = server.start()
            api = server
            emit_api(**st)
        except Exception as e:  # noqa: BLE001
            emit_api(False, note="не удалось запустить API", error=str(e))

    def poll_stop():
        while True:
            try:
                raw = commands.get_nowait()
            except queue.Empty:
                return
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if obj.get("cmd") in ("stop", "shutdown"):
                stop_flag.set()

    while True:
        raw = commands.get()
        stop_flag.clear()
        try:
            req = json.loads(raw)
        except Exception:
            continue
        cmd = req.get("cmd")

        if cmd == "shutdown":
            stop_api(announce=False)
            break

        if cmd == "load":
            cfg = req.get("api") or {}
            if "enabled" in cfg:
                api_enabled = bool(cfg.get("enabled"))
            if "port" in cfg:
                try:
                    api_port = int(cfg.get("port") or 8000)
                except (TypeError, ValueError):
                    api_port = 8000
            if cfg.get("host"):
                api_host = str(cfg.get("host"))
            stop_api(announce=False)
            with backend_lock:
                if backend is not None:
                    backend.close()
                    backend = None
            model = req.get("model") or ""
            adapter = req.get("adapter") or ""
            threads = int(req.get("threads") or os.cpu_count() or 4)
            ctx = int(req.get("ctx") or CTX_DEFAULT)
            try:
                kv_type = req.get("kv_type") or "f16"
                with backend_lock:
                    if model.lower().endswith(".gguf"):
                        backend = GgufBackend(model, threads, ctx, kv_type)
                        backend_kind = "gguf"
                    else:
                        backend = TransformersBackend(model, adapter, threads)
                        backend_kind = "transformers"
                    info = backend.start()
                emit({"ev": "ready", "backend": backend_kind, "info": info})
                if api_enabled:
                    start_api(api_enabled, api_port, api_host)
            except Exception as e:  # noqa: BLE001
                emit({"ev": "error", "message": str(e)})
                backend = None
                emit_api(False, note="API остановлен: модель не загрузилась")

        elif cmd == "api":
            if "enabled" in req:
                api_enabled = bool(req.get("enabled"))
            if "port" in req:
                try:
                    api_port = int(req.get("port") or 8000)
                except (TypeError, ValueError):
                    api_port = 8000
            if req.get("host"):
                api_host = str(req.get("host"))
            start_api(api_enabled, api_port, api_host)

        elif cmd == "agent":
            if backend is None or backend_kind != "gguf":
                emit({"ev": "error", "message": "Агент работает только на GGUF-модели (llama.cpp)."})
                continue
            try:
                with backend_lock:
                    steps = backend.run_agent(
                        req.get("task") or "",
                        req.get("root") or ".",
                        int(req.get("max_steps") or 30),
                        int(req.get("max_tokens") or 2048),
                        float(req.get("temp") or 0.3),
                        emit,
                        stop_flag.is_set,
                        think=bool(req.get("think")),
                        web=bool(req.get("web")),
                        web_engine=req.get("web_engine") or "duckduckgo",
                    )
                emit({"ev": "agent_done", "steps": steps})
            except Exception as e:  # noqa: BLE001
                emit({"ev": "error", "message": str(e)})

        elif cmd == "chat":
            if backend is None:
                emit({"ev": "error", "message": "Сначала загрузи модель"})
                continue
            messages = req.get("messages") or []
            if not messages or messages[0].get("role") != "system":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
            if req.get("web"):
                engine = req.get("web_engine") or "duckduckgo"
                topic = ""
                for m in reversed(messages):
                    if m.get("role") == "user":
                        topic = m.get("content") or ""
                        break
                if topic:
                    emit({"ev": "web", "engine": engine, "query": topic, "found": 0})
                    results, err = search_web(topic, engine)
                    emit({"ev": "web", "engine": engine, "query": topic,
                          "found": len(results), "note": err})
                    if results:
                        emit({"ev": "web", "engine": engine, "query": topic,
                              "found": len(results), "note": "читаю верхние страницы…"})
                    note = (web_context(topic, engine, results, fetch_pages=True)
                            if results
                            else "Веб-поиск (%s) не дал результатов: %s" % (engine, err))
                    messages = messages[:1] + [{"role": "system", "content": note}] + messages[1:]
            max_tokens = int(req.get("max_tokens") or 256)
            temp = float(req.get("temp") or 0.7)
            top_p = float(req.get("top_p") or 0.95)
            think = bool(req.get("think"))
            counter = {"n": 0}

            def on_token(text):
                counter["n"] += 1
                emit({"ev": "token", "text": text})
                if counter["n"] % 8 == 0:
                    poll_stop()

            def on_reasoning(text):
                emit({"ev": "reasoning", "text": text})

            try:
                with backend_lock:
                    tokens, elapsed = backend.chat(messages, max_tokens, temp, top_p,
                                                   on_token, stop_flag.is_set,
                                                   think=think, on_reasoning=on_reasoning)
                emit({"ev": "done", "tokens": int(tokens),
                      "tok_s": round(tokens / elapsed, 2) if elapsed else 0.0,
                      "backend": backend_kind})
            except Exception as e:  # noqa: BLE001
                emit({"ev": "error", "message": str(e)})

    stop_api(announce=False)
    if backend is not None:
        backend.close()


if __name__ == "__main__":
    main()
