"""Поиск, скачивание и предпросмотр моделей/датасетов с Hugging Face.

Каждая команда — отдельный процесс, вывод — JSON-строки:
  python hf_tools.py search  model|dataset <query>
  python hf_tools.py download model|dataset <repo> <dest>
  python hf_tools.py preview dataset <repo>
"""

import json
import os
import re
import sys
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

MODEL_EXT = (".safetensors", ".bin", ".gguf", ".json", ".model", ".txt", ".jinja", ".tiktoken")
DATA_EXT = (".jsonl", ".json", ".csv", ".parquet", ".arrow", ".txt")


def out(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def api():
    from huggingface_hub import HfApi
    return HfApi()


def _as_dt(value):
    if not value:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value


def ago(value):
    dt = _as_dt(value)
    if not dt:
        return ""
    secs = max(0, (datetime.now(timezone.utc) - dt).total_seconds())
    if secs < 3600:
        return "%d мин" % max(1, secs // 60)
    if secs < 86400:
        return "%d ч" % (secs // 3600)
    if secs < 86400 * 30:
        return "%d дн" % (secs // 86400)
    if secs < 86400 * 365:
        return "%d мес" % (secs // (86400 * 30))
    return "%d г" % (secs // (86400 * 365))


def is_fresh(value, days=7):
    dt = _as_dt(value)
    if not dt:
        return False
    return (datetime.now(timezone.utc) - dt).total_seconds() < days * 86400


def date_str(value):
    dt = _as_dt(value)
    if not dt:
        return ""
    months = ["янв", "фев", "мар", "апр", "мая", "июн",
              "июл", "авг", "сен", "окт", "ноя", "дек"]
    return "%d %s %d" % (dt.day, months[dt.month - 1], dt.year)


def quant_label(filename):
    m = re.search(r"((?:UD-)?(?:IQ\d[A-Za-z0-9_]*|Q\d[A-Za-z0-9_]*|BF16|F16|F32))",
                  os.path.basename(filename), re.I)
    return m.group(1).upper() if m else ""


def item_json(it):
    mid = getattr(it, "id", "") or ""
    author = mid.split("/")[0]
    tags = [t for t in (getattr(it, "tags", None) or [])
            if not t.startswith(("license:", "autotrain", "region:"))][:4]
    modified = getattr(it, "last_modified", None) or getattr(it, "created_at", None)
    return {
        "type": "item",
        "id": mid,
        "downloads": int(getattr(it, "downloads", 0) or 0),
        "likes": int(getattr(it, "likes", 0) or 0),
        "tags": tags,
        "author": author,
        "updated": ago(modified),
        "created": date_str(getattr(it, "created_at", None) or modified),
        "new": is_fresh(modified),
        "library": getattr(it, "library_name", None) or "",
    }


def list_models(a, **kw):
    try:
        return list(a.list_models(direction=-1, **kw))
    except TypeError:
        return list(a.list_models(**kw))


def fit_quant(a, repo, budget_gb):
    """Лучший по качеству GGUF-квант, который влезает в бюджет."""
    try:
        info = a.model_info(repo, files_metadata=True)
    except Exception:
        return None
    limit = max(0.5, budget_gb) * 1e9
    # Сплит-кванты лежат шардами: складываем размеры одной группы.
    groups = {}
    for s in info.siblings or []:
        name = s.rfilename or ""
        low = name.lower()
        if not low.endswith(".gguf"):
            continue
        if any(x in low for x in ("mmproj", "projector", "vision",
                                  "dspark", "imatrix")):
            continue
        quant = quant_label(name)
        if not quant:
            continue
        size = int(s.size or 0)
        key = (os.path.dirname(name), quant)
        groups.setdefault(key, []).append(size)
    fits = []
    for (_, quant), sizes in groups.items():
        if not sizes or any(sz <= 0 for sz in sizes):
            continue
        total = sum(sizes)
        if total <= limit:
            fits.append((quant, total))
    if not fits:
        return None
    # Лучшее качество, которое ещё влезает.
    quant, size = max(fits, key=lambda x: x[1])
    return {"quant": quant, "file_size": size}


def do_feed(user, quants, ram_gb, cpu):
    a = api()
    seen = set()

    def section(title, items, note=""):
        fresh = []
        for it in items:
            key = (it.get("id") or "").lower()
            if not key or key in seen:
                continue
            seen.add(key)
            fresh.append(it)
        if not fresh:
            return
        out({"type": "section", "title": title, "note": note})
        for it in fresh:
            out(it)
        out({"type": "section_end"})

    try:
        latest = list_gguf(a, 24, author=user, sort="createdAt")[:3]
        section("Свежие релизы " + user, [item_json(it) for it in latest],
                "автор с галочкой")
    except Exception as e:  # noqa: BLE001
        out({"type": "log", "message": "Релизы %s: %s" % (user, e)})

    try:
        try:
            pool = list_models(a, author="unsloth", sort="createdAt",
                               limit=40, filter="gguf")
        except Exception:
            pool = list_models(a, author="unsloth", sort="createdAt", limit=60)
            pool = [p for p in pool
                    if "gguf" in (p.id or "").lower()
                    or any("gguf" in str(t).lower() for t in (getattr(p, "tags", None) or []))]
        if quants == "runnable":
            chosen = []
            tries = 0
            budget = min(max(1.0, ram_gb) * 0.9,
                         max(1, cpu) * 1.25)
            for it in pool:
                if len(chosen) >= 3:
                    break
                tries += 1
                if tries > 16:
                    break
                extra = fit_quant(a, it.id, budget)
                if extra:
                    item = item_json(it)
                    item.update(extra)
                    chosen.append(item)
            section("Кванты под твоё железо", chosen,
                    "до %.0f ГБ · %d потоков" % (budget, max(1, cpu)))
        else:
            section("Свежие кванты Unsloth",
                    [item_json(it) for it in pool[:3]],
                    "последние GGUF")
    except Exception as e:  # noqa: BLE001
        out({"type": "log", "message": "Кванты Unsloth: %s" % e})

    try:
        popular = list_gguf(a, 40, sort="downloads")[:8]
        section("Популярное на Hugging Face",
                [item_json(it) for it in popular], "по скачиваниям")
    except Exception as e:  # noqa: BLE001
        out({"type": "error", "message": "Популярное: %s" % e})

    out({"type": "done"})


def is_gguf(it):
    mid = (getattr(it, "id", "") or "").lower()
    tags = [str(t).lower() for t in (getattr(it, "tags", None) or [])]
    return "gguf" in mid or any("gguf" in t for t in tags)


def list_gguf(a, limit, **kw):
    """Только GGUF-репозитории: их файлы реально скачиваются и запускаются."""
    try:
        items = list_models(a, filter="gguf", limit=limit, **kw)
    except Exception:
        items = list_models(a, limit=limit, **kw)
    return [it for it in items if is_gguf(it)]


def do_author(user, sort="downloads"):
    a = api()
    try:
        items = list_gguf(a, 100, author=user, sort=sort)
    except Exception as e:  # noqa: BLE001
        out({"type": "error", "message": "Профиль %s: %s" % (user, e)})
        return
    out({"type": "profile", "author": user, "count": len(items)})
    for it in items:
        out(item_json(it))
    out({"type": "done"})


def do_search(kind, query, limit=12, sort="downloads"):
    a = api()
    try:
        # В новых huggingface_hub параметра direction уже нет.
        def fetch():
            fn = a.list_datasets if kind == "dataset" else a.list_models
            kw = {"search": query or None, "sort": sort, "limit": limit}
            if kind == "model":
                kw["filter"] = "gguf"
            try:
                return list(fn(direction=-1, **kw))
            except TypeError:
                return list(fn(**kw))

        items = fetch()
        if kind == "model":
            items = [it for it in items if is_gguf(it)]
    except Exception as e:  # noqa: BLE001
        out({"type": "error", "message": "Поиск не удался: %s" % e})
        return
    for it in items:
        out(item_json(it))
    out({"type": "done", "count": len(items)})


def repo_files(kind, repo):
    a = api()
    info = (a.dataset_info(repo, files_metadata=True) if kind == "dataset"
            else a.model_info(repo, files_metadata=True))
    files = []
    for s in info.siblings or []:
        name = s.rfilename
        if name.startswith(".git"):
            continue
        ext = os.path.splitext(name)[1].lower()
        allow = DATA_EXT if kind == "dataset" else MODEL_EXT
        if ext in allow:
            files.append((name, int(s.size or 0)))
    return files


def gguf_variants(files):
    """Склеиваем шарды в один квант: UD-Q4_K_XL, Q8_0 и т.д."""
    groups = {}
    for name, size in files:
        low = name.lower()
        if not low.endswith(".gguf"):
            continue
        if any(x in low for x in ("mmproj", "projector", "vision",
                                  "dspark", "imatrix")):
            continue
        label = quant_label(name)
        if not label:
            continue
        folder = os.path.basename(os.path.dirname(name))
        if folder and re.search(r"(?i)^(ud-)?(iq\d|q\d)", folder):
            label = folder
        label = label.upper()
        groups.setdefault((os.path.dirname(name), label), []).append((name, int(size or 0)))
    from collections import Counter
    counts = Counter(label for (_, label) in groups)
    variants = []
    for (dirname, label), shards in groups.items():
        parent = os.path.basename(dirname)
        shown = label
        if counts[label] > 1 and parent:
            shown = "%s · %s" % (parent, label)
        variants.append({
            "label": shown,
            "size": sum(sz for _, sz in shards),
            "files": [n for n, _ in shards],
        })
    variants.sort(key=lambda v: v["size"])
    return variants


def do_files(kind, repo):
    try:
        files = repo_files(kind, repo)
    except Exception as e:  # noqa: BLE001
        out({"type": "error", "message": "Не удалось получить список файлов: %s" % e})
        return
    payload = {"type": "files", "items": [{"name": n, "size": sz} for n, sz in files]}
    if kind == "model":
        variants = gguf_variants(files)
        if variants:
            payload["variants"] = variants
    out(payload)


def do_download(kind, repo, dest, only=None, quant=None):
    try:
        from huggingface_hub import hf_hub_download
        files = repo_files(kind, repo)
        if quant:
            variants = {v["label"]: set(v["files"]) for v in gguf_variants(files)}
            wanted = variants.get(quant)
            if not wanted:
                out({"type": "error", "message": "Квант %s не найден" % quant})
                return
            files = [f for f in files if f[0] in wanted]
        if only:
            files = [f for f in files if f[0] == only]
            if not files:
                out({"type": "error", "message": "Файл %s не найден в репозитории" % only})
                return
    except Exception as e:  # noqa: BLE001
        out({"type": "error", "message": "Не удалось получить список файлов: %s" % e})
        return
    if not files:
        out({"type": "error", "message": "В репозитории нет подходящих файлов"})
        return
    total = sum(size for _, size in files) or 1
    done = 0
    os.makedirs(dest, exist_ok=True)
    for name, size in files:
        out({"type": "progress", "file": name, "done": done, "total": total})
        try:
            hf_hub_download(repo_id=repo, filename=name, repo_type=kind,
                            local_dir=dest)
        except Exception as e:  # noqa: BLE001
            out({"type": "error", "message": "Ошибка на %s: %s" % (name, e)})
            return
        done += size
        out({"type": "progress", "file": name, "done": done, "total": total})
    out({"type": "done", "path": dest})


def plain_markdown(text):
    """Убираем markdown-разметку: заголовки, ссылки, картинки, акценты."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.M)
    text = re.sub(r"\*\*([^*]+)\*\*", r"", text)
    text = re.sub(r"`([^`]+)`", r"", text)
    return text


def do_readme(kind, repo):
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename="README.md", repo_type=kind)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read(6000)
        # YAML-шапка карточки модели не нужна в предпросмотре
        if text.startswith("---"):
            end = text.find("\n---", 3)
            if end != -1:
                text = text[end + 4:]
        out({"type": "readme", "text": text})
    except Exception as e:  # noqa: BLE001
        out({"type": "error", "message": "README недоступен: %s" % e})


def do_preview(kind, repo, rows=3):
    try:
        files = repo_files(kind, repo)
    except Exception as e:  # noqa: BLE001
        out({"type": "error", "message": "Не удалось получить список файлов: %s" % e})
        return
    data = [f for f, _ in files if os.path.splitext(f)[1].lower() in (".jsonl", ".json", ".csv", ".parquet")]
    # jsonl читается построчно и надёжнее — ставим его первым
    data.sort(key=lambda f: 0 if f.endswith(".jsonl") else 1)
    if not data:
        out({"type": "error", "message": "В датасете нет jsonl/json/csv/parquet"})
        return
    name = data[0]
    try:
        import fsspec
        path = "hf://datasets/%s/%s" % (repo, name)
        with fsspec.open(path, "rb", block_size=0) as fh:
            head = fh.read(192 * 1024)
        text = head.decode("utf-8", "replace")
        if name.endswith((".jsonl", ".json")):
            items = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
                if len(items) >= rows:
                    break
            if not items:
                # JSON-массив: вытаскиваем первые объекты по балансу скобок,
                # даже если голова файла обрезана.
                depth, start = 0, None
                for i, ch in enumerate(text):
                    if ch == "{":
                        if depth == 0:
                            start = i
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0 and start is not None:
                            try:
                                items.append(json.loads(text[start:i + 1]))
                            except Exception:
                                pass
                            start = None
                            if len(items) >= rows:
                                break
            out({"type": "preview", "file": name, "rows": items})
        elif name.endswith(".csv"):
            import csv, io
            reader = csv.DictReader(io.StringIO(text))
            items = []
            for row in reader:
                items.append(dict(row))
                if len(items) >= rows:
                    break
            out({"type": "preview", "file": name, "rows": items})
        else:
            out({"type": "error", "message": "Предпросмотр parquet не поддержан — файл: %s" % name})
    except Exception as e:  # noqa: BLE001
        out({"type": "error", "message": "Предпросмотр не удался: %s" % e})


def main():
    if len(sys.argv) < 2:
        out({"type": "error", "message": "Неверные аргументы"})
        return
    cmd = sys.argv[1]
    if cmd == "author":
        user = sys.argv[2] if len(sys.argv) > 2 else ""
        sort = sys.argv[3] if len(sys.argv) > 3 else "downloads"
        do_author(user, sort)
        return
    if cmd == "feed":
        user = "YoSaNoo"
        quants = "unsloth"
        ram = 6.5
        cpu = 8
        args = sys.argv[2:]
        for i, arg in enumerate(args):
            if arg == "--user" and i + 1 < len(args):
                user = args[i + 1]
            elif arg == "--quants" and i + 1 < len(args):
                quants = args[i + 1]
            elif arg == "--ram" and i + 1 < len(args):
                try:
                    ram = float(args[i + 1])
                except Exception:
                    pass
            elif arg == "--cpu" and i + 1 < len(args):
                try:
                    cpu = int(args[i + 1])
                except Exception:
                    pass
        do_feed(user, quants, ram, cpu)
        return
    if len(sys.argv) < 3:
        out({"type": "error", "message": "Неверные аргументы"})
        return
    kind = sys.argv[2]
    if cmd == "search":
        sort = sys.argv[4] if len(sys.argv) > 4 else "downloads"
        do_search(kind, sys.argv[3] if len(sys.argv) > 3 else "", 12, sort)
    elif cmd == "files":
        do_files(kind, sys.argv[3])
    elif cmd == "download":
        rest = sys.argv[5:]
        only = None
        quant = None
        if rest and rest[0] == "--quant" and len(rest) > 1:
            quant = rest[1]
        elif rest:
            only = rest[0]
        do_download(kind, sys.argv[3], sys.argv[4], only, quant)
    elif cmd == "preview":
        do_preview(kind, sys.argv[3])
    elif cmd == "readme":
        do_readme(kind, sys.argv[3])
    else:
        out({"type": "error", "message": "Неизвестная команда: %s" % cmd})


if __name__ == "__main__":
    main()
