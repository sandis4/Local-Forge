"""
Interactive chat with real-time streaming and tok/s counter.
Works with ANY trained model:
  - LoRA adapter on top of a base:  python test_chat.py --base models/gemma_base --lora output_model/final_model
  - Merged model directly:          python test_chat.py --base models/lf_e2b_merged --no_lora
  - Any architecture (Gemma/Qwen/...): big models auto-load via mmap if RAM is tight.
"""

import os
import sys
import json
import time
import argparse
import glob
import multiprocessing
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

from config import DEFAULT_MODEL_PATH
from model_loader import load_model_auto

TOTAL_CPUS = multiprocessing.cpu_count()
torch.set_num_threads(TOTAL_CPUS)
try:
    torch.set_num_interop_threads(max(1, min(4, TOTAL_CPUS // 4)))
except RuntimeError:
    pass

def find_latest_checkpoint(output_dir: str = "output_model") -> str:
    candidates = [
        os.path.join(output_dir, "final_model"),
        os.path.join(output_dir, "interactive_checkpoint"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "adapter_model.safetensors")) or os.path.exists(os.path.join(c, "adapter_config.json")):
            return c

    step_dirs = glob.glob(os.path.join(output_dir, "checkpoint-step-*"))
    if step_dirs:
        step_dirs.sort(key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0, reverse=True)
        return step_dirs[0]

    return os.path.join(output_dir, "final_model")

# Модельно-независимая личность (без упоминания архитектуры — подходит и Gemma, и Qwen)
FORGE_SYSTEM_PROMPT = (
    "You are Forge, an intelligent and helpful AI assistant created and fine-tuned by "
    "the Reforge Squad team using the Fable-5-Premium dataset and SWE-smith software "
    "engineering data. Always answer in the user's language."
)

STOP_TOKEN_STRINGS = ("<end_of_turn>", "<|im_end|>", "<|eot_id|>", "<turn|>", "<|end|>")

def collect_eos_ids(tokenizer):
    ids = []
    if tokenizer.eos_token_id is not None:
        ids.append(tokenizer.eos_token_id)
    for s in STOP_TOKEN_STRINGS:
        try:
            tid = tokenizer.convert_tokens_to_ids(s)
            if isinstance(tid, int) and tid >= 0 and tid not in ids:
                ids.append(tid)
        except Exception:
            pass
    return ids or None

try:
    import colorama
    colorama.init()
    DIM, RESET = colorama.Style.DIM, colorama.Style.RESET_ALL
except Exception:
    DIM, RESET = "", ""

class ThinkSplitPrinter:
    """Режет поток генерации на «мысли» и ответ.
    Мысли (<think>...</think> у Qwen, <|channel>thought...<channel|> у Gemma)
    печатаются приглушённо, в историю идёт только чистый ответ.
    in_think=True — если шаблон УЖЕ открыл блок мыслей в промпте (Qwen так делает)."""

    STARTS = ("<think>", "<|channel>thought")
    ENDS = ("</think>", "<channel|>")
    STOPS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end_of_text|>",
             "<|endoftext|>", "<|return|>", "<turn|>")
    KEEP = 24  # хвост-буфер на случай маркера, разрезанного между чанками

    def __init__(self, in_think: bool = False):
        self.in_think = in_think
        self._header_shown = False
        self.stopped = False
        self.buf = ""
        self.answer = ""
        self.think = ""

    def _ensure_think_header(self):
        if not self._header_shown:
            print(f"\n{DIM}[💭 мысли] {RESET}", end="", flush=True)
            self._header_shown = True

    def _find(self, markers):
        best, bm = None, None
        for m in markers:
            i = self.buf.find(m)
            if i >= 0 and (best is None or i < best):
                best, bm = i, m
        return best, bm

    def _safe_flush(self):
        if len(self.buf) <= self.KEEP:
            return ""
        out, self.buf = self.buf[:-self.KEEP], self.buf[-self.KEEP:]
        return out

    def feed(self, piece):
        if self.stopped or not piece:
            return
        if self.in_think:
            self._ensure_think_header()
        self.buf += piece
        while True:
            if self.in_think:
                idx, m = self._find(self.ENDS)
                if idx is None:
                    flush = self._safe_flush()
                    if flush:
                        self.think += flush
                        print(DIM + flush + RESET, end="", flush=True)
                    return
                flush = self.buf[:idx]
                self.think += flush
                if flush:
                    print(DIM + flush + RESET, end="", flush=True)
                self.buf = self.buf[idx + len(m):]
                self.in_think = False
                print("\n[💭 конец мыслей]\nОтвет > ", end="", flush=True)
            else:
                sidx, _ = self._find(self.STOPS)
                if sidx is not None:
                    txt = self.buf[:sidx]
                    self.answer += txt
                    print(txt, end="", flush=True)
                    self.buf = ""
                    self.stopped = True
                    return
                idx, m = self._find(self.STARTS)
                if idx is None:
                    flush = self._safe_flush()
                    if flush:
                        self.answer += flush
                        print(flush, end="", flush=True)
                    return
                txt = self.buf[:idx]
                self.answer += txt
                if txt.strip():
                    print(txt, end="", flush=True)
                self.buf = self.buf[idx + len(m):]
                self.in_think = True
                self._ensure_think_header()

    def finish(self):
        if self.stopped or not self.buf:
            self.buf = ""
            return
        if self.in_think:
            self.think += self.buf
            print(DIM + self.buf + RESET, end="", flush=True)
        else:
            self.answer += self.buf
            print(self.buf, end="", flush=True)
        self.buf = ""

def trim_history(tokenizer, history, system_prompt, max_ctx_tokens: int):
    """Обрезает старые реплики, если история перестала влезать в контекст."""
    def total_len(h):
        enc = tokenizer.apply_chat_template(h, tokenize=True, return_dict=True)
        return len(enc["input_ids"])

    while len(history) > 3 and total_len(history) > max_ctx_tokens:
        # удаляем самую старую пару user+assistant (system [0] не трогаем)
        del history[1]
        if len(history) > 1 and history[1].get("role") == "assistant":
            del history[1]
    return history

def adapter_matches_base(lora_path: str, base_model_path: str):
    """True/False: записана ли в адаптере та же база. None = не удалось прочитать."""
    cfg_path = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return None
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        recorded = os.path.normpath(os.path.abspath(cfg.get("base_model_name_or_path") or ""))
        base = os.path.normpath(os.path.abspath(base_model_path))
        return recorded == base
    except Exception:
        return None

def check_tokenizer_model_match(tokenizer, model):
    """Ловит скрещивание токенизатора с чужой моделью (источник мусора <unusedN>)."""
    tv = len(tokenizer)
    cfg = getattr(model, "config", None)
    mv = getattr(cfg, "vocab_size", None)
    if not mv:
        mv = getattr(getattr(cfg, "text_config", None), "vocab_size", None)
    if mv and abs(tv - mv) > 500:
        raise SystemExit(
            f"\n[ОШИБКА] Токенизатор ({tv} токенов) не подходит модели ({mv} токенов)!\n"
            f"  Скорее всего LoRA от другой архитектуры. Запусти с --no_lora\n"
            f"  или укажи адаптер, обученный на этой базе.\n"
        )

def prompt_preopened_think(prompt_text: str) -> bool:
    """True, если chat-шаблон уже открыл блок мыслей в промпте и не закрыл его
    (Qwen вставляет '<think>\n' после assistant — тогда модель закрывает его сама)."""
    return (prompt_text.count("<think>") > prompt_text.count("</think>")) or \
           (prompt_text.count("<|channel>thought") > prompt_text.count("<channel|>"))

def chat_loop(base_model_path: str, lora_weights_path: str = None, use_lora: bool = True,
              max_tokens: int = 2048, threads: int = TOTAL_CPUS, system_prompt: str = FORGE_SYSTEM_PROMPT,
              temperature: float = 0.7, repetition_penalty: float = 1.15):
    if threads:
        torch.set_num_threads(threads)

    if use_lora and (not lora_weights_path or not os.path.exists(os.path.join(lora_weights_path, "adapter_config.json"))):
        found = find_latest_checkpoint("output_model")
        if os.path.exists(os.path.join(found, "adapter_config.json")):
            lora_weights_path = found
        else:
            lora_weights_path = None

    # Защита от скрещивания: адаптер от другой базы молча ломает генерацию в мусор
    if use_lora and lora_weights_path:
        match = adapter_matches_base(lora_weights_path, base_model_path)
        if match is False:
            print(f"[ВНИМАНИЕ] LoRA '{lora_weights_path}' обучена на ДРУГОЙ базе — отключаю её, чат на чистой модели.")
            lora_weights_path = None

    mode = f"LoRA: {lora_weights_path}" if (use_lora and lora_weights_path) else "слитая модель (без LoRA)"
    print("=" * 60)
    print(" ЗАГРУЗКА FORGE...")
    print(f" Модель: {base_model_path}")
    print(f" Режим: {mode}")
    print(f" Потоков CPU: {threads} | Лимит ответа: {max_tokens} токенов")
    print("=" * 60)

    tokenizer_path = base_model_path
    if use_lora and lora_weights_path and os.path.exists(os.path.join(lora_weights_path, "tokenizer_config.json")):
        tokenizer_path = lora_weights_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("Загрузка модели (auto: RAM или mmap для больших)...")
    model, load_info = load_model_auto(base_model_path, gradient_checkpointing=False)
    src = "mmap (с диска)" if load_info["mmap"] else "RAM"
    print(f"Готово: {load_info['weights_gb']:.1f} ГБ, {load_info['dtype']}, источник: {src}")

    if use_lora and lora_weights_path:
        print(f"Подключение LoRA весов из '{lora_weights_path}'...")
        try:
            peft_model = PeftModel.from_pretrained(model, lora_weights_path)
            model = peft_model.merge_and_unload()
            print("LoRA веса успешно объединены с базой (дообучение активно)!")
        except Exception as e:
            print(f"Слияние не удалось ({e}), подключение в стандартном режиме...")
            model = PeftModel.from_pretrained(model, lora_weights_path)
    elif use_lora:
        print("LoRA чекпоинт не найден — чат со слитой/чистой моделью.")

    model.eval()

    # Санити: словарь токенизатора должен совпадать со словарём модели
    check_tokenizer_model_match(tokenizer, model)

    eos_ids = collect_eos_ids(tokenizer)
    max_ctx = int(getattr(getattr(model, "config", None), "max_position_embeddings", 8192) or 8192)
    max_ctx = min(max_ctx, 8192)

    print("\n" + "=" * 60)
    print(" Forge готов к диалогу! Команды: exit | /reset | /system <промпт> | /think on|off")
    print("=" * 60 + "\n")

    history = []
    if system_prompt:
        history.append({"role": "system", "content": system_prompt})

    # Режим мышления: None = как решит сама модель (её дефолт), True/False — принудительно
    think_kw = None

    while True:
        try:
            user_input = input("User > ")
            if not user_input.strip():
                continue
            low = user_input.strip().lower()
            if low in ["exit", "quit"]:
                break
            if low == "/reset":
                history = [{"role": "system", "content": system_prompt}] if system_prompt else []
                print("[История очищена]\n")
                continue
            if low.startswith("/system"):
                new_sys = user_input[7:].strip()
                system_prompt = new_sys or FORGE_SYSTEM_PROMPT
                history = [{"role": "system", "content": system_prompt}]
                print(f"[Системный промпт обновлён]\n")
                continue
            if low == "/think" or low.startswith("/think "):
                arg = low[6:].strip()
                if arg in ("on", "вкл", "1", "true"):
                    think_kw = True
                elif arg in ("off", "выкл", "0", "false"):
                    think_kw = False
                elif arg in ("auto", "авто", ""):
                    think_kw = None
                state = {True: "всегда думать", False: "без мыслей", None: "на усмотрение модели"}[think_kw]
                print(f"[Режим мышления: {state}]\n")
                continue

            history.append({"role": "user", "content": user_input})
            history = trim_history(tokenizer, history, system_prompt, max_ctx - max_tokens)

            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                tmpl_kwargs = {}
                if think_kw is not None:
                    tmpl_kwargs["enable_thinking"] = think_kw
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        history, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
                    )
                except Exception:
                    prompt_text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
            else:
                prompt_text = f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"

            inputs = tokenizer(prompt_text, return_tensors="pt")

            # Спец-токены НЕ скрываем: по маркерам <think>/<|channel>thought отделяем мысли.
            # Если шаблон уже открыл <think> в промпте — стартуем сразу "в мыслях"
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
            printer = ThinkSplitPrinter(in_think=prompt_preopened_think(prompt_text))

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )

            print("\nAssistant > ", end="", flush=True)

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            start_time = time.perf_counter()
            token_count = 0

            for new_text in streamer:
                printer.feed(new_text)
                token_count += 1

            thread.join()
            printer.finish()
            full_response = printer.answer.strip()
            elapsed_time = max(0.001, time.perf_counter() - start_time)

            total_gen_tokens = token_count if token_count else len(tokenizer.encode(full_response, add_special_tokens=False))
            tps = total_gen_tokens / elapsed_time

            print(f"\n\n[⚡ {total_gen_tokens} токенов за {elapsed_time:.2f} сек | Скорость: {tps:.2f} tok/s]")
            if printer.think:
                print(f"[💭 мысли заняли {len(tokenizer.encode(printer.think, add_special_tokens=False))} токенов — в историю не попали]")
            print()

            history.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\n")
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default=DEFAULT_MODEL_PATH, help="Model folder (base or merged)")
    parser.add_argument("--lora", type=str, default="output_model/final_model", help="Path to saved LoRA adapter folder")
    parser.add_argument("--no_lora", action="store_true", help="Chat with base/merged model directly, skip LoRA")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max output tokens")
    parser.add_argument("--threads", type=int, default=TOTAL_CPUS, help="Number of CPU threads to use")
    parser.add_argument("--system", type=str, default=FORGE_SYSTEM_PROMPT, help="System prompt ('' = disable)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    args = parser.parse_args()

    chat_loop(args.base, args.lora, use_lora=not args.no_lora, max_tokens=args.max_tokens,
              threads=args.threads, system_prompt=args.system, temperature=args.temperature,
              repetition_penalty=args.repetition_penalty)
