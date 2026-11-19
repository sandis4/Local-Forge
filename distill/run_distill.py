"""Дистилляция DeepSeek Flash по банку вопросов.

Запуск (когда появится баланс):
    set DEEPSEEK_API_KEY=sk-...
    python run_distill.py --questions questions_3000.jsonl --dry-run
    python run_distill.py --questions questions_3000.jsonl --concurrency 16

Официальные параметры: https://api-docs.deepseek.com/
"""

import argparse
import gzip
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

DEFAULT_SYSTEM = "You are Local Forge, a helpful AI assistant. Answer in the user language."

PRICES = {
    "peak": {"hit": 0.006, "miss": 0.30, "out": 1.20},
    "off": {"hit": 0.003, "miss": 0.15, "out": 0.60},
}


class FatalApiError(RuntimeError):
    pass


def now_is_peak():
    now = datetime.now(timezone.utc)
    return now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)


def cost_usd(hit, miss, out, mode):
    p = PRICES[mode]
    return (hit * p["hit"] + miss * p["miss"] + out * p["out"]) / 1_000_000.0


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_questions(path):
    rows = read_jsonl(path)
    if not rows:
        raise SystemExit("Нет вопросов в %s" % path)
    return rows


def build_body(q, args):
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.api_system},
            {"role": "user", "content": q["prompt"]},
        ],
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    if args.logprobs > 0:
        body["logprobs"] = True
        body["top_logprobs"] = args.logprobs
    if args.thinking == "off":
        body["thinking"] = {"type": "disabled"}
        body["temperature"] = args.temperature
    else:
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = args.thinking
    return body


def compact_tokens(entries, top_k):
    out = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        top = []
        for t in (e.get("top_logprobs") or [])[:top_k]:
            if isinstance(t, dict) and t.get("token") is not None:
                top.append([t.get("token"), round(float(t.get("logprob", 0.0)), 5)])
        lp = e.get("logprob")
        out.append({
            "t": e.get("token", ""),
            "lp": round(float(lp), 5) if lp is not None else None,
            "top": top,
        })
    return out


def teacher_summary(entries):
    entries = entries or []
    vals = [float(e["logprob"]) for e in entries
            if isinstance(e, dict) and isinstance(e.get("logprob"), (int, float))
            and float(e["logprob"]) > -9000]
    if not vals:
        return {}
    mean = sum(vals) / len(vals)
    return {
        "tokens": len(entries),
        "scored": len(vals),
        "mean_logprob": round(mean, 4),
        "perplexity": round(math.exp(-mean), 4) if -mean < 50 else None,
        "outside_topk": sum(1 for e in entries
                            if isinstance(e, dict) and isinstance(e.get("logprob"), (int, float))
                            and float(e["logprob"]) <= -9000),
    }


def call_api(body, args, key):
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + key,
            "User-Agent": "local-forge-distill/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    if not data.get("choices"):
        raise RuntimeError("пустой ответ API: %s" % str(data)[:300])
    choice = data["choices"][0]
    msg = choice.get("message") or {}
    content = (msg.get("content") or "").strip()
    reasoning = (msg.get("reasoning_content") or "").strip()
    if not content:
        raise RuntimeError("модель вернула пустой ответ (finish_reason=%s)" % choice.get("finish_reason"))
    return content, reasoning, choice.get("finish_reason"), data.get("usage") or {}, choice.get("logprobs")


def run_one(q, args, key, abort):
    for attempt in range(args.retries + 1):
        if abort.is_set():
            return None
        try:
            content, reasoning, finish, usage, raw_lp = call_api(build_body(q, args), args, key)
            row = {
                "id": q["id"],
                "prompt": q["prompt"],
                "lang": q.get("lang", "ru"),
                "domain": q.get("domain", ""),
                "kind": q.get("kind", ""),
                "difficulty": q.get("difficulty", ""),
                "model": args.model,
                "thinking": args.thinking,
                "reasoning": reasoning,
                "answer": content,
                "finish_reason": finish,
                "usage": usage,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            soft = None
            if args.logprobs > 0 and isinstance(raw_lp, dict):
                answer_lp = compact_tokens(raw_lp.get("content"), args.logprobs)
                reasoning_lp = compact_tokens(raw_lp.get("reasoning_content"), args.logprobs)
                row["teacher"] = {
                    "top_k": args.logprobs,
                    "answer": teacher_summary(raw_lp.get("content")),
                    "reasoning": teacher_summary(raw_lp.get("reasoning_content")),
                }
                soft = {"id": q["id"], "answer": answer_lp}
                if args.logprob_scope == "all":
                    soft["reasoning"] = reasoning_lp
            return row, soft
        except urllib.error.HTTPError as e:
            try:
                detail = e.read().decode("utf-8", "replace")[:800]
            except Exception:
                detail = ""
            if e.code in (400, 401, 402, 422):
                abort.set()
                raise FatalApiError("HTTP %d: %s" % (e.code, detail)) from e
            if attempt >= args.retries:
                raise RuntimeError("HTTP %d после %d попыток: %s" % (e.code, attempt + 1, detail)) from e
            delay = min(60.0, (2 ** attempt) + random.random())
            print("  HTTP %d, повтор через %.0f c" % (e.code, delay), file=sys.stderr)
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
            if attempt >= args.retries:
                raise RuntimeError("сетевая ошибка после %d попыток: %s" % (attempt + 1, e)) from e
            delay = min(60.0, (2 ** attempt) + random.random())
            print("  сеть: %s, повтор через %.0f c" % (e, delay), file=sys.stderr)
            time.sleep(delay)
    return None


def use_think(a, target, max_length):
    """Оставлять ли reasoning в обучающем тексте.

    Reasoning DeepSeek часто длиннее всего контекста (медиана ~2.4K символов),
    поэтому в «auto» он попадает в данные только если заведомо влезает вместе
    с финальным ответом. Иначе учим только ответ — обрезанный ответ хуже,
    чем ответ без рассуждения.
    """
    if not a.get("reasoning") or target == "answer":
        return False
    if target == "think":
        return True
    chars = len(a["reasoning"]) + len(a["answer"])
    return chars <= max(200, int(max_length * 2.5 * 0.75))


def answer_likely_too_long(answer, max_length):
    # Оценка снизу по символам: русский BPE ~2.2–3 символа на токен.
    return len(answer or "") > max_length * 2.2


def convert_to_training(questions, answers_path, training_path, system,
                        target="auto", max_length=512):
    answers = {}
    for row in read_jsonl(answers_path):
        if row.get("answer"):
            answers[row["id"]] = row
    n = 0
    thinks = 0
    too_long = 0
    with open(training_path, "w", encoding="utf-8") as f:
        for q in questions:
            a = answers.get(q["id"])
            if not a:
                continue
            think = use_think(a, target, max_length)
            thinks += int(think)
            too_long += int(answer_likely_too_long(a["answer"], max_length))
            row = training_row(q, a, system, think)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print("Обучение: %s (%d диалогов, с reasoning %d, длинных ответов ~%d)"
          % (os.path.abspath(training_path), n, thinks, too_long))
    return n


def training_row(q, a, system, think=False):
    text = a["answer"]
    if think and a.get("reasoning"):
        text = "<think>\n%s\n</think>\n\n%s" % (a["reasoning"], text)
    return {
        "id": q["id"],
        "domain": q.get("domain", ""),
        "source": "deepseek-flash",
        "teacher": a.get("teacher"),
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": q["prompt"]},
            {"role": "assistant", "content": text},
        ],
    }


def build_kd(answers_path, soft_path, kd_path, system, ids=None,
             target="auto", max_length=512):
    """Собирает KD-датасет из готовых ответов и файла шансов.

    Читаем soft потоково, поэтому 15K вопросов не требуют держать все
    распределения в RAM. Файл перезаписывается целиком — докачка не плодит
    дубликаты.
    """
    answers = {r["id"]: r for r in read_jsonl(answers_path) if r.get("answer")}
    n = 0
    thinks = 0
    too_long = 0
    with gzip.open(soft_path, "rt", encoding="utf-8") as sf, \
            gzip.open(kd_path, "wt", encoding="utf-8") as kf:
        for line in sf:
            line = line.strip()
            if not line:
                continue
            try:
                soft = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ids is not None and soft.get("id") not in ids:
                continue
            a = answers.get(soft.get("id"))
            if not a:
                continue
            q = {
                "id": a["id"],
                "prompt": a["prompt"],
                "domain": a.get("domain", ""),
                "kind": a.get("kind", ""),
                "difficulty": a.get("difficulty", ""),
            }
            think = use_think(a, target, max_length)
            thinks += int(think)
            too_long += int(answer_likely_too_long(a["answer"], max_length))
            row = training_row(q, a, system, think)
            if think:
                row["soft"] = soft
            else:
                row["soft"] = {"answer": soft.get("answer") or []}
            kf.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print("KD-датасет:      %s (%d диалогов, с reasoning %d, длинных ответов ~%d)"
          % (os.path.abspath(kd_path), n, thinks, too_long))
    return n


def estimate(questions, args):
    prompt_chars = sum(len(q["prompt"]) for q in questions)
    est_in = max(1, prompt_chars // 3 + 50 * len(questions))
    per_q = args.est_output + (args.est_reasoning if args.thinking != "off" else 0)
    est_out = per_q * len(questions)
    print("Вопросов:        %d" % len(questions))
    print("Средний промпт:  %.0f символов" % (prompt_chars / max(1, len(questions))))
    print("Оценка токенов:  вход ~%.2fM, выход ~%.2fM" % (est_in / 1e6, est_out / 1e6))
    if args.logprobs > 0:
        print("Logprobs:        top-%d, scope=%s (на цену токенов не влияет)" % (args.logprobs, args.logprob_scope))
    else:
        print("Logprobs:        выключены")
    print("Обучение:        target=%s, max_length=%d%s"
          % (args.train_target, args.train_max_length,
             ", ответ просят короче" if args.max_answer_tokens > 0 else ""))
    for mode, name in (("off", "off-peak"), ("peak", "peak")):
        print("Оценка цены (%s): ~$%.2f" % (name, cost_usd(0, est_in, est_out, mode)))
    print("Сейчас: %s. Реальные токены появятся после запуска." % ("PEAK" if now_is_peak() else "OFF-PEAK"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="questions_3000.jsonl")
    ap.add_argument("--out", default="answers_3000.jsonl")
    ap.add_argument("--training-out", default="train_distill.jsonl")
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--thinking", choices=["off", "low", "high", "max"], default="high")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--system-file", default="")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--convert-only", action="store_true")
    ap.add_argument("--est-output", type=int, default=700, help="средний ответ, токенов")
    ap.add_argument("--est-reasoning", type=int, default=1400, help="среднее рассуждение, токенов")
    ap.add_argument("--logprobs", type=int, default=5, help="top_logprobs 0..20; 0 = не запрашивать")
    ap.add_argument("--logprob-scope", choices=["answer", "all"], default="answer")
    ap.add_argument("--logprobs-out", default="", help="gzip JSONL; по умолчанию <out>.logprobs.jsonl.gz")
    ap.add_argument("--kd-out", default="", help="KD-датасет с soft; по умолчанию <training_out>.kd.jsonl.gz")
    ap.add_argument("--train-target", choices=["auto", "answer", "think"], default="auto",
                    help="что класть в ассистента: только ответ, ответ+reasoning, или auto по бюджету")
    ap.add_argument("--train-max-length", type=int, default=512,
                    help="контекст, под который подгоняется auto (как в Local Forge)")
    ap.add_argument("--max-answer-tokens", type=int, default=0,
                    help="если >0 — попросить учителя отвечать короче (в промпт API, не в обучение)")
    args = ap.parse_args()
    if not 0 <= args.logprobs <= 20:
        raise SystemExit("--logprobs должен быть от 0 до 20")
    if not args.logprobs_out:
        args.logprobs_out = args.out + ".logprobs.jsonl.gz"
    if not args.kd_out:
        base = args.training_out
        if base.endswith(".jsonl.gz"):
            base = base[:-9]
        elif base.endswith(".jsonl"):
            base = base[:-6]
        args.kd_out = base + ".kd.jsonl.gz"

    args.system = DEFAULT_SYSTEM
    if args.system_file:
        with open(args.system_file, encoding="utf-8") as f:
            args.system = f.read().strip()
    args.api_system = args.system
    if args.max_answer_tokens > 0:
        args.api_system += (
            " Keep the final answer complete but concise: under %d tokens. "
            "Do not repeat the question or pad the answer." % args.max_answer_tokens
        )

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]

    if args.convert_only:
        convert_to_training(questions, args.out, args.training_out, args.system, target=args.train_target, max_length=args.train_max_length)
        if args.logprobs > 0 and os.path.exists(args.logprobs_out):
            build_kd(args.out, args.logprobs_out, args.kd_out, args.system, ids={q["id"] for q in questions}, target=args.train_target, max_length=args.train_max_length)
        return

    if args.dry_run:
        estimate(questions, args)
        return

    key = os.environ.get(args.key_env)
    if not key:
        raise SystemExit("Нет ключа: задай переменную окружения %s" % args.key_env)

    done = {r["id"] for r in read_jsonl(args.out) if r.get("answer")}
    todo = [q for q in questions if q["id"] not in done]
    print("Всего: %d, уже готово: %d, осталось: %d" % (len(questions), len(done), len(todo)))
    log_info = ("top-%d/%s" % (args.logprobs, args.logprob_scope)) if args.logprobs else "off"
    print("Модель: %s, thinking=%s, logprobs=%s, параллельно: %d"
          % (args.model, args.thinking, log_info, args.concurrency))
    if not todo:
        convert_to_training(questions, args.out, args.training_out, args.system, target=args.train_target, max_length=args.train_max_length)
        if args.logprobs > 0 and os.path.exists(args.logprobs_out):
            build_kd(args.out, args.logprobs_out, args.kd_out, args.system, ids={q["id"] for q in questions}, target=args.train_target, max_length=args.train_max_length)
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    abort = threading.Event()
    totals = {"hit": 0, "miss": 0, "out": 0, "n": 0}
    confidence = []
    failures = []

    def account(row):
        u = row.get("usage") or {}
        hit = int(u.get("prompt_cache_hit_tokens") or 0)
        miss = int(u.get("prompt_cache_miss_tokens") or 0)
        prompt_tokens = int(u.get("prompt_tokens") or 0)
        if not miss:
            miss = max(0, prompt_tokens - hit)
        out = int(u.get("completion_tokens") or 0)
        totals["hit"] += hit
        totals["miss"] += miss
        totals["out"] += out
        totals["n"] += 1

    t0 = time.time()
    soft_sink = gzip.open(args.logprobs_out, "at", encoding="utf-8") if args.logprobs > 0 else None
    try:
        with open(args.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(run_one, q, args, key, abort): q for q in todo}
            for fut in as_completed(futures):
                q = futures[fut]
                try:
                    result = fut.result()
                except FatalApiError as e:
                    failures.append((q["id"], str(e)))
                    print("ФАТАЛЬНО: %s" % e, file=sys.stderr)
                    break
                except Exception as e:
                    failures.append((q["id"], str(e)))
                    print("ОШИБКА %s: %s" % (q["id"], e), file=sys.stderr)
                    continue
                if not result:
                    continue
                row, soft = result
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
                if soft and soft_sink:
                    soft_sink.write(json.dumps(soft, ensure_ascii=False) + "\n")
                account(row)
                t = (row.get("teacher") or {}).get("answer") or {}
                if "mean_logprob" in t:
                    confidence.append(float(t["mean_logprob"]))
                if totals["n"] % 25 == 0:
                    elapsed = time.time() - t0
                    rate = totals["n"] / max(1e-9, elapsed)
                    if soft_sink:
                        soft_sink.flush()
                    print("  готово %d/%d, %.1f вопр/с, ошибок %d" % (totals["n"], len(todo), rate, len(failures)))
            if abort.is_set():
                for fut in futures:
                    fut.cancel()
    finally:
        if soft_sink:
            soft_sink.close()

    elapsed = time.time() - t0
    print("Готово за %.0f c. Успешно: %d, ошибок: %d" % (elapsed, totals["n"], len(failures)))
    if totals["n"]:
        hit, miss, out = totals["hit"], totals["miss"], totals["out"]
        print("Токены: вход %d (cache hit %d / miss %d), выход %d" % (hit + miss, hit, miss, out))
        for mode, name in (("off", "off-peak"), ("peak", "peak")):
            print("Оценка цены (%s): $%.2f" % (name, cost_usd(hit, miss, out, mode)))
    if confidence:
        mean_lp = sum(confidence) / len(confidence)
        ppl = math.exp(-mean_lp) if -mean_lp < 50 else float("inf")
        print("Уверенность учителя: logprob %.3f, perplexity ~%.1f (%d ответов)"
              % (mean_lp, ppl, len(confidence)))
    if args.logprobs > 0:
        print("Шансы по токенам: %s" % os.path.abspath(args.logprobs_out))
    if failures:
        err_path = args.out + ".errors.jsonl"
        with open(err_path, "w", encoding="utf-8") as f:
            for qid, msg in failures:
                f.write(json.dumps({"id": qid, "error": msg}, ensure_ascii=False) + "\n")
        print("Ошибки: %s (перезапуск докачает их)" % os.path.abspath(err_path))

    convert_to_training(questions, args.out, args.training_out, args.system, target=args.train_target, max_length=args.train_max_length)
    if args.logprobs > 0 and os.path.exists(args.logprobs_out):
        build_kd(args.out, args.logprobs_out, args.kd_out, args.system, ids={q["id"] for q in questions}, target=args.train_target, max_length=args.train_max_length)


if __name__ == "__main__":
    main()
