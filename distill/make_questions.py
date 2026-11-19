"""Материализует банк вопросов в JSONL.

python make_questions.py --count 3000 --out questions_3000.jsonl
"""

import argparse
import json
import os
from collections import Counter

from question_bank import build_master


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3000)
    ap.add_argument("--out", default="questions_3000.jsonl")
    args = ap.parse_args()

    master, by_domain = build_master()
    if args.count > len(master):
        raise SystemExit("В банке только %d вопросов (запрошено %d)" % (len(master), args.count))
    rows = master[:args.count]

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for q in rows:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    domains = Counter(q["domain"] for q in rows)
    kinds = Counter(q["kind"] for q in rows)
    diff = Counter(q["difficulty"] for q in rows)
    print("Банк: %d вопросов (запас %d)" % (len(master), len(master) - len(rows)))
    print("Записано: %s" % os.path.abspath(args.out))
    print("По доменам: " + ", ".join("%s=%d" % (k, v) for k, v in domains.most_common()))
    print("По типам:   " + ", ".join("%s=%d" % (k, v) for k, v in kinds.most_common(12)))
    print("Сложность:  " + ", ".join("%s=%d" % (k, v) for k, v in diff.most_common()))


if __name__ == "__main__":
    main()
