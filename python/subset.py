"""
Сборщик учебной подвыборки: из всего database/ делает один плотный train_11.jsonl.
Зачем: весь датасет (13.5 ГБ, ~800k строк) на CPU — это год; отобранные
~250-300k сильных строк на бесплатной GPU (Kaggle/Colab) — недели.

  --src file[:cap] — источники по приоритету (повторы файла учитываются 1 раз)
  --cap N          — всего строк на выходе
  --max-tokens N   — режем монстров (оценка локальным токенизатором, иначе len//4)
  --min-chars N    — выбрасываем пустышки без ответа
  --out PATH       — куда писать

Дедуп глобальный: sha1 по (последний user + последний assistant, в нижнем регистре).
Отчёт в конце: взято/отброшено по источникам + оценка шагов эпохи при accum 8/32.
"""

import argparse
import hashlib
import json
import os
import sys

TOKENIZER = None


def est_tokens(text: str) -> int:
    global TOKENIZER
    if TOKENIZER is not None:
        try:
            return len(TOKENIZER.encode(text, add_special_tokens=False))
        except Exception:
            pass
    return max(1, len(text) // 4)


def load_tokenizer():
    global TOKENIZER
    for cand in ("models/Qwen-3.5-9B", "models/gemma_base"):
        try:
            from transformers import AutoTokenizer
            TOKENIZER = AutoTokenizer.from_pretrained(cand, trust_remote_code=True)
            print(f"[Токенизатор] точная оценка длины: {cand}")
            return
        except Exception:
            continue
    print("[Токенизатор] локального нет — грубая оценка len//4")


def row_texts(row: dict):
    """Возвращает (prompt_tail, answer_tail) для хэша и фильтров."""
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        users = [m.get("content", "") for m in msgs
                 if str(m.get("role", "")).lower() in ("user", "human")]
        assts = []
        for m in msgs:
            if str(m.get("role", "")).lower() in ("assistant", "model", "bot"):
                c = m.get("content") or ""
                if not str(c).strip() and m.get("tool_calls"):
                    # агентские датасеты: ответ — это вызовы инструментов
                    calls = []
                    for tc in (m["tool_calls"] if isinstance(m["tool_calls"], list)
                               else [m["tool_calls"]]):
                        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                        calls.append(f"[Call: {fn.get('name', 'call')}({fn.get('arguments', '')})]")
                    c = "\n".join(calls)
                assts.append(c if isinstance(c, str) else json.dumps(c, ensure_ascii=False))
        return (users[-1] if users else ""), (assts[-1] if assts else "")
    for pk in ("prompt", "instruction", "input"):
        if row.get(pk):
            p = row[pk]
            p = p if isinstance(p, str) else json.dumps(p, ensure_ascii=False)
            for ak in ("chosen", "rejected", "response", "output", "answer", "completion"):
                if row.get(ak):
                    a = row[ak]
                    a = a if isinstance(a, str) else json.dumps(a, ensure_ascii=False)
                    return p, a
    s = json.dumps(row, ensure_ascii=False)
    return s[:2000], s[-2000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", nargs="+", required=True,
                    help="файлы [file[:cap] ...] по приоритету")
    ap.add_argument("--cap", type=int, default=300000)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--out", default="database/train_11.jsonl")
    args = ap.parse_args()

    load_tokenizer()
    seen_hashes = set()
    total_kept = 0
    out_f = open(args.out, "w", encoding="utf-8")
    try:
        for spec in args.src:
            if ":" in spec and not spec.endswith(".jsonl"):
                path, cap_s = spec.rsplit(":", 1)
                try:
                    cap = int(cap_s)
                except ValueError:
                    path, cap = spec, None
            else:
                path, cap = spec, None
            if not os.path.exists(path):
                print(f"[SKIP] нет файла: {path}")
                continue
            kept = drop_dup = drop_short = drop_long = drop_bad = 0
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if total_kept >= args.cap or (cap is not None and kept >= cap):
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        drop_bad += 1
                        continue
                    if not isinstance(row, dict):
                        drop_bad += 1
                        continue
                    pr, an = row_texts(row)
                    if not an or len(an.strip()) < args.min_chars:
                        drop_short += 1
                        continue
                    h = hashlib.sha1(
                        ((pr.strip() + "\n" + an.strip()).lower()).encode(
                            "utf-8", "ignore")).hexdigest()
                    if h in seen_hashes:
                        drop_dup += 1
                        continue
                    if est_tokens(pr) + est_tokens(an) > args.max_tokens:
                        drop_long += 1
                        continue
                    seen_hashes.add(h)
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
                    total_kept += 1
            print(f"[{os.path.basename(path)}] взято {kept}, дубли {drop_dup}, "
                  f"короткие {drop_short}, длинные {drop_long}, битые {drop_bad}")
            if total_kept >= args.cap:
                break
    finally:
        out_f.close()
    est = int(total_kept * 1.15)  # +identity-подмес
    print("=" * 60)
    print(f"ИТОГО {total_kept} строк -> {args.out}")
    print(f"Шагов эпохи: ~{est // 8} при accum 8 | ~{est // 32} при accum 32")
    print("=" * 60)


if __name__ == "__main__":
    main()
