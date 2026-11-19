"""
Dataset with weighted mixing of the main database and identity reinforcement samples.
Ensures Forge identity is consistently embedded alongside the full database knowledge.
"""

import gzip
import json
import math
import os
import re
import random
import torch
from torch.utils.data import IterableDataset
from typing import Iterator, Dict, Any, Optional

import hashlib


def _open_text(path: str):
    if str(path).lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def _cache_slug(path: str) -> str:
    return hashlib.md5(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]


class StreamingConversationDataset(IterableDataset):
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Any,
        identity_path: str = "database/identity.jsonl",
        identity_mix_ratio: float = 0.15, # 15% of samples injected with identity Q&A
        max_length: int = 512,
        max_samples: Optional[int] = None,
        train_only_on_assistant: bool = True,
        cache_dir: Optional[str] = None,  # кэш токенизации: скип 748 жирных сэмплов станет секундами
        start_line: int = 0,              # мгновенный скип: начать с этой строки файла
        kd: bool = False,                 # дистилляция: soft-targets учителя из поля soft
        kd_top_k: int = 8,
    ):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.identity_path = identity_path
        self.identity_mix_ratio = identity_mix_ratio
        self.max_length = max_length
        self.max_samples = max_samples
        self.train_only_on_assistant = train_only_on_assistant
        self.cache_dir = cache_dir
        self.kd = bool(kd)
        self.kd_top_k = max(1, int(kd_top_k))
        self._kd_token_cache = {}
        self._kd_warn_strip = False
        self._kd_warn_skip = False
        self.cache_hits = 0
        self.cache_miss = 0
        self.start_line = max(0, int(start_line or 0))
        self.current_line = self.start_line
        self._cache_ok_memo = {}

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"File not found: {jsonl_path}")

        self.identity_samples = []
        if os.path.exists(identity_path):
            with open(identity_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        self.identity_samples.append(json.loads(line))
                    except Exception:
                        pass
            print(f"Loaded {len(self.identity_samples)} identity reinforcement samples.")

    def _normalize_role(self, role: str) -> str:
        r = str(role).lower()
        if r in ["system", "sys"]:
            return "system"
        elif r in ["user", "human"]:
            return "user"
        elif r in ["assistant", "model", "bot"]:
            return "assistant"
        elif r in ["tool", "observation", "code_interpreter"]:
            return "tool"
        return "user"

    @staticmethod
    def _assistant_pattern():
        return re.compile(
            r'(<\|turn>model\n|<start_of_turn>model\n|<\|im_start\|>assistant\n|<\|assistant\|>\n'
            r'|<\|start_header_id\|>assistant<\|end_header_id\|>\n\n'
            r'|<\|message\|>assistant\n)'
            r'(.*?)'
            r'(<turn\|>|<end_of_turn>|<\|im_end\|>|<\|eot_id\|>|<\|end\|>|<turn\|>\n|$)',
            re.DOTALL
        )

    def _render_text(self, formatted_messages):
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            try:
                return self.tokenizer.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
            except Exception:
                pass
        return self._build_fallback_text(formatted_messages)

    def _tokenize_text(self, full_text):
        try:
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,
                return_tensors="pt"
            )
            return (encoding["input_ids"][0], encoding["attention_mask"][0],
                    encoding["offset_mapping"][0], True)
        except Exception:
            encoding = self.tokenizer(
                full_text,
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )
            return encoding["input_ids"][0], encoding["attention_mask"][0], None, False

    def _assistant_truncated(self, full_text, offset_mapping):
        if offset_mapping is None or len(offset_mapping) == 0:
            return True
        covered = int(offset_mapping[:, 1].max().item())
        matches = list(self._assistant_pattern().finditer(full_text))
        if not matches:
            return False
        return matches[-1].end(2) > covered

    @staticmethod
    def _strip_think_from_last_assistant(formatted_messages):
        for m in reversed(formatted_messages):
            if m.get("role") != "assistant":
                continue
            text = m.get("content") or ""
            stripped = re.sub(r"<think>.*?</think>\s*", "", text,
                              flags=re.DOTALL | re.IGNORECASE)
            if stripped != text:
                m["content"] = stripped
                return True
            break
        return False

    def _process_item(self, data: dict) -> Optional[Dict[str, torch.Tensor]]:
        messages = data.get("messages", [])
        # Новые срезы Fable хранят messages как JSON-строку, а не список.
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except Exception:
                return None
        if isinstance(messages, dict):
            messages = messages.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return None

        formatted_messages = []
        has_system = any(
            isinstance(m, dict) and self._normalize_role(m.get("role", "")) == "system"
            for m in messages
        )

        # Prepend default identity system prompt if missing
        if not has_system:
            formatted_messages.append({
                "role": "system",
            "content": "You are Local Forge, a helpful AI assistant. Answer in the user language."
        })

        for m in messages:
            if not isinstance(m, dict):
                continue
            role = self._normalize_role(m.get("role", "user"))
            content = m.get("content") or ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)

            # tool_calls могут идти вместе с текстом ассистента — сохраняем
            # аргументы (пути/команды), иначе модель учится только выбору инструмента.
            tool_calls = m.get("tool_calls")
            if tool_calls:
                tc_list = tool_calls if isinstance(tool_calls, list) else [tool_calls]
                calls_str = []
                for tc in tc_list:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", "call")
                    args = fn.get("arguments", "")
                    if not isinstance(args, str):
                        args = json.dumps(args, ensure_ascii=False)
                    calls_str.append(f"[Call: {name}({args})]")
                calls_text = "\n".join(calls_str)
                content = (content.rstrip() + "\n" + calls_text).strip() if content.strip() else calls_text

            if role == "tool":
                formatted_messages.append({
                    "role": "user",
                    "content": f"[Tool Output]:\n{content}"
                })
            else:
                formatted_messages.append({
                    "role": role,
                    "content": content
                })

        # Apply chat template
        input_ids = attention_mask = offset_mapping = None
        for attempt in range(2):
            full_text = self._render_text(formatted_messages)
            input_ids, attention_mask, offset_mapping, _ = self._tokenize_text(full_text)
            if not self.kd or not self._assistant_truncated(full_text, offset_mapping):
                break
            if attempt == 0 and self._strip_think_from_last_assistant(formatted_messages):
                if not self._kd_warn_strip:
                    print("[KD] Ассистент не влез в max_length — оставляю финальный ответ без reasoning.")
                    self._kd_warn_strip = True
                continue
            if not self._kd_warn_skip:
                print(f"[KD] Сэмпл длиннее max_length={self.max_length}: пропускаю, "
                      f"чтобы не учиться на обрезанном ответе. Увеличь контекст или сократи ответ.")
                self._kd_warn_skip = True
            return None

        if not self.train_only_on_assistant or offset_mapping is None:
            labels = input_ids.clone()
        else:
            labels = torch.full_like(input_ids, -100)
            pattern = re.compile(
                r'(<\|turn>model\n|<start_of_turn>model\n|<\|im_start\|>assistant\n|<\|assistant\|>\n'
                r'|<\|start_header_id\|>assistant<\|end_header_id\|>\n\n'
                r'|<\|message\|>assistant\n)'
                r'(.*?)'
                r'(<turn\|>|<end_of_turn>|<\|im_end\|>|<\|eot_id\|>|<\|end\|>|<turn\|>\n|$)',
                re.DOTALL
            )
            for match in pattern.finditer(full_text):
                start_char = match.start(2)
                end_char = match.end()
                for idx, (token_start, token_end) in enumerate(offset_mapping):
                    if token_start >= start_char and token_end <= end_char:
                        labels[idx] = input_ids[idx]

            if (labels != -100).sum().item() < 2:
                # Если в сэмпле нет токенов ассистента — бракуем его,
                # но НЕ включаем labels = input_ids.clone(), иначе модель выучит системные промпты и user-реплики!
                return None

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        if self.kd:
            out.update(self._build_soft(data, full_text, offset_mapping, input_ids.size(0)))
        return out

    def _zero_soft(self, length: int):
        return {
            "soft_ids": torch.zeros((length, self.kd_top_k), dtype=torch.long),
            "soft_probs": torch.zeros((length, self.kd_top_k), dtype=torch.float32),
            "soft_mask": torch.zeros((length,), dtype=torch.float32),
        }

    def _teacher_first_id(self, text: str):
        if text in self._kd_token_cache:
            return self._kd_token_cache[text]
        sid = None
        if text:
            try:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if ids:
                    sid = int(ids[0])
            except Exception:
                sid = None
        self._kd_token_cache[text] = sid
        return sid

    def _teacher_probs(self, entry: dict):
        """Проекция top-k учителя на первый токен ученика.

        Токенизаторы разные: один токен учителя может распадаться на несколько
        токенов ученика. Поэтому вероятность альтернативы складывается в первый
        токен ученика, который получается при её токенизации. Это приближение
        сохраняет все top-k и даёт корректное распределение по next-token.
        """
        pairs = []
        for item in (entry.get("top") or []):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((item[0], item[1]))
        if entry.get("t") is not None and entry.get("lp") is not None:
            pairs.append((entry.get("t"), entry.get("lp")))

        acc = {}
        seen = set()
        for tok, lp in pairs:
            if not tok or tok in seen:
                continue
            seen.add(tok)
            try:
                lp = float(lp)
            except (TypeError, ValueError):
                continue
            if lp <= -9998:
                continue
            sid = self._teacher_first_id(str(tok))
            if sid is None:
                continue
            acc[sid] = acc.get(sid, 0.0) + math.exp(lp)

        if len(acc) < 2:
            return [], []
        total = sum(acc.values())
        if total <= 0:
            return [], []
        items = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[:self.kd_top_k]
        keep = sum(p for _, p in items)
        if keep <= 0:
            return [], []
        return [i for i, _ in items], [p / keep for _, p in items]

    def _build_soft(self, data, full_text, offset_mapping, length):
        zero = self._zero_soft(length)
        soft = data.get("soft")
        if not self.kd or not isinstance(soft, dict) or offset_mapping is None:
            return zero

        pattern = re.compile(
            r'(<\|turn>model\n|<start_of_turn>model\n|<\|im_start\|>assistant\n|<\|assistant\|>\n'
            r'|<\|start_header_id\|>assistant<\|end_header_id\|>\n\n'
            r'|<\|message\|>assistant\n)'
            r'(.*?)'
            r'(<turn\|>|<end_of_turn>|<\|im_end\|>|<\|eot_id\|>|<\|end\|>|<turn\|>\n|$)',
            re.DOTALL
        )
        spans = [(m.start(2), m.end(2)) for m in pattern.finditer(full_text)]
        if not spans:
            return zero
        span_start, span_end = spans[-1]
        assistant_text = full_text[span_start:span_end]

        by_start = {}
        for key in ("reasoning", "answer"):
            entries = soft.get(key) or []
            if not entries:
                continue
            section = "".join(str(e.get("t", "")) for e in entries)
            if not section:
                continue
            pos = assistant_text.find(section)
            if pos < 0:
                continue
            cursor = span_start + pos
            for e in entries:
                t = str(e.get("t", ""))
                if not t:
                    continue
                by_start.setdefault(cursor, e)
                cursor += len(t)

        if not by_start:
            return zero

        ids_out = torch.zeros((length, self.kd_top_k), dtype=torch.long)
        probs_out = torch.zeros((length, self.kd_top_k), dtype=torch.float32)
        mask_out = torch.zeros((length,), dtype=torch.float32)
        hits = 0
        for idx, (token_start, token_end) in enumerate(offset_mapping):
            if idx >= length:
                break
            ts, te = int(token_start), int(token_end)
            if ts < span_start or te > span_end:
                continue
            entry = by_start.get(ts)
            if not entry:
                continue
            ids, probs = self._teacher_probs(entry)
            if not ids:
                continue
            n = min(len(ids), self.kd_top_k)
            ids_out[idx, :n] = torch.tensor(ids[:n], dtype=torch.long)
            probs_out[idx, :n] = torch.tensor(probs[:n], dtype=torch.float32)
            mask_out[idx] = 1.0
            hits += 1
        if hits == 0:
            return zero
        return {"soft_ids": ids_out, "soft_probs": probs_out, "soft_mask": mask_out}

    def _build_fallback_text(self, formatted_messages) -> str:
        full_text = ""
        for m in formatted_messages:
            role = m["role"]
            if role == "assistant":
                role = "model"
            full_text += f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n"
        return full_text

    def _cache_ok(self, src: str) -> bool:
        """Кэш валиден, только если исходник не менялся (mtime+size+max_length).
        Мемоизация: раньше вызывался на КАЖДЫЙ сэмпл (2 файловых операции),
        из-за чего скип тысяч сэмплов занимал минуты."""
        if src in self._cache_ok_memo:
            return self._cache_ok_memo[src]
        try:
            if not self.cache_dir:
                self._cache_ok_memo[src] = False
                return False
            with open(os.path.join(self.cache_dir, _cache_slug(src), "meta.json"),
                      encoding="utf-8") as f:
                meta = json.load(f)
            st = os.stat(src)
            ok = (meta.get("mtime_ns") == st.st_mtime_ns
                  and meta.get("size") == st.st_size
                  and meta.get("max_length") == self.max_length
                  and meta.get("kd", False) == bool(self.kd)
                  and meta.get("kd_top_k", 8) == self.kd_top_k)
        except Exception:
            ok = False
        self._cache_ok_memo[src] = ok
        return ok

    def _cache_get(self, src: str, idx: int):
        try:
            p = os.path.join(self.cache_dir, _cache_slug(src), f"{idx}.pt")
            if os.path.exists(p):
                self.cache_hits += 1
                return torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            pass
        self.cache_miss += 1
        return None

    def _cache_put(self, src: str, idx: int, sample: dict):
        try:
            d = os.path.join(self.cache_dir, _cache_slug(src))
            os.makedirs(d, exist_ok=True)
            st = os.stat(src)
            with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
                json.dump({"mtime_ns": st.st_mtime_ns, "size": st.st_size,
                           "max_length": self.max_length,
                           "kd": bool(self.kd), "kd_top_k": self.kd_top_k}, f)
            torch.save({k: v.cpu() for k, v in sample.items()},
                       os.path.join(d, f"{idx}.pt"))
        except Exception:
            pass

    def _cached_or_process(self, src: str, idx: int, data: dict):
        """Токенизация дорогая (секунды на жирных строках): второй проход
        и все скипы берут готовые тензоры с диска."""
        if self.cache_dir and self._cache_ok(src):
            hit = self._cache_get(src, idx)
            if hit is not None:
                return hit
        out = self._process_item(data)
        if out is not None and self.cache_dir:
            self._cache_put(src, idx, out)
        return out

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        yield_count = 0
        with _open_text(self.jsonl_path) as f:
            for line_idx, line in enumerate(f):
                # Мгновенный скип: просто читаем строки, БЕЗ json/токенизации/кэша.
                # Раньше «пропуск 5600 батчей» тянул 5600 кэш-файлов = 10 минут.
                if line_idx < self.start_line:
                    continue
                if self.max_samples and yield_count >= self.max_samples:
                    break
                self.current_line = line_idx

                # Interleave identity samples to embed identity permanently.
                # RNG-последовательность НЕ меняем: randrange = один _randbelow,
                # как раньше делал choice (иначе съедет точное продолжение).
                if self.identity_samples and random.random() < self.identity_mix_ratio:
                    id_idx = random.randrange(len(self.identity_samples))
                    parsed_id = self._cached_or_process(
                        self.identity_path, id_idx, self.identity_samples[id_idx])
                    if parsed_id is not None:
                        yield_count += 1
                        yield parsed_id
                        if self.max_samples and yield_count >= self.max_samples:
                            break

                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                if self.cache_dir:
                    sample = self._cached_or_process(self.jsonl_path, line_idx, data)
                else:
                    sample = self._process_item(data)
                if sample is not None:
                    yield_count += 1
                    yield sample


class ReplayMixedDataset(IterableDataset):
    """
    Anti-forgetting replay: к основному потоку сэмплов подмешивает с вероятностью
    replay_ratio сэмплы из общего (репетиционного) датасета. Модель во время узкого
    дообучения продолжает "репетировать" общие знания и не забывает их.
    """

    def __init__(self, main_dataset, replay_dataset, replay_ratio: float = 0.2, seed: int = 42):
        super().__init__()
        self.main = main_dataset
        self.replay = replay_dataset
        self.replay_ratio = replay_ratio
        self.seed = seed
        # Персистентный RNG (не пересоздаётся каждую эпоху): репетиционные
        # сэмплы не повторяются от эпохи к эпохе, а состояние можно
        # сохранить/восстановить для точного продолжения обучения.
        self._rng = random.Random(seed)

    def get_rng_state(self):
        return self._rng.getstate()

    def set_rng_state(self, state):
        self._rng.setstate(state)

    def __iter__(self):
        main_iter = iter(self.main)
        replay_iter = iter(self.replay)
        rng = self._rng
        replay_exhausted = False
        for item in main_iter:
            yield item
            if not replay_exhausted and rng.random() < self.replay_ratio:
                try:
                    yield next(replay_iter)
                except StopIteration:
                    replay_exhausted = True


def collate_fn(batch, pad_token_id: int = 0) -> Dict[str, torch.Tensor]:
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        l = item["input_ids"].size(0)
        pad_len = max_len - l
        if pad_len > 0:
            pad_ids = torch.full((pad_len,), pad_token_id, dtype=torch.long)
            pad_mask = torch.zeros((pad_len,), dtype=torch.long)
            pad_labels = torch.full((pad_len,), -100, dtype=torch.long)

            input_ids.append(torch.cat([item["input_ids"], pad_ids]))
            attention_mask.append(torch.cat([item["attention_mask"], pad_mask]))
            labels.append(torch.cat([item["labels"], pad_labels]))
        else:
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
            labels.append(item["labels"])

    out = {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels)
    }
    # Packing кладёт в сэмпл готовые окна одинаковой длины + 4D-маску между
    # документами. Без packing этих ключей в сэмпле нет.
    if batch and "position_ids" in batch[0]:
        out["position_ids"] = torch.stack([item["position_ids"] for item in batch])
    if batch and "block_mask" in batch[0]:
        out["block_mask"] = torch.stack([item["block_mask"] for item in batch])

    soft_items = [item for item in batch if "soft_ids" in item]
    if soft_items:
        k = max(item["soft_ids"].size(-1) for item in soft_items)
        s_ids, s_probs, s_mask = [], [], []
        for item in batch:
            l = item["input_ids"].size(0)
            pad_len = max_len - l
            if "soft_ids" in item:
                si = item["soft_ids"]
                sp = item["soft_probs"]
                sm = item["soft_mask"]
                if si.size(-1) < k:
                    si = torch.cat([si, torch.zeros((l, k - si.size(-1)), dtype=torch.long)], -1)
                    sp = torch.cat([sp, torch.zeros((l, k - sp.size(-1)), dtype=torch.float32)], -1)
            else:
                si = torch.zeros((l, k), dtype=torch.long)
                sp = torch.zeros((l, k), dtype=torch.float32)
                sm = torch.zeros((l,), dtype=torch.float32)
            if pad_len > 0:
                si = torch.cat([si, torch.zeros((pad_len, k), dtype=torch.long)])
                sp = torch.cat([sp, torch.zeros((pad_len, k), dtype=torch.float32)])
                sm = torch.cat([sm, torch.zeros((pad_len,), dtype=torch.float32)])
            s_ids.append(si)
            s_probs.append(sp)
            s_mask.append(sm)
        out["soft_ids"] = torch.stack(s_ids)
        out["soft_probs"] = torch.stack(s_probs)
        out["soft_mask"] = torch.stack(s_mask)
    return out


class PreferenceDataset(IterableDataset):
    """Датасет пар (chosen / rejected) для ORPO.

    Строка jsonl — один из форматов:
      {"messages_chosen": [...], "messages_rejected": [...]}
      {"prompt": str|messages, "chosen": str|messages, "rejected": str|messages}
    Токенизация и маскировка промпта (-100) — те же, что в SFT
    (переиспользуем StreamingConversationDataset._process_item).
    """

    def __init__(self, jsonl_path: str, tokenizer: Any,
                 max_length: int = 512, max_samples: Optional[int] = None,
                 cache_dir: Optional[str] = None):
        super().__init__()
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"File not found: {jsonl_path}")
        self.jsonl_path = jsonl_path
        self.max_samples = max_samples
        # Форматтер: тот же файл как jsonl_path (существует), identity выкл
        self._fmt = StreamingConversationDataset(
            jsonl_path=jsonl_path, tokenizer=tokenizer,
            identity_path="", identity_mix_ratio=0.0,
            max_length=max_length, max_samples=None,
            train_only_on_assistant=True, cache_dir=cache_dir,
        )

    @staticmethod
    def _to_messages(part) -> list:
        if isinstance(part, list):
            return part
        return [{"role": "user", "content": "Ответь на запрос."},
                {"role": "assistant", "content": str(part)}]

    def _row_to_pair(self, row: dict):
        if "messages_chosen" in row or "messages_rejected" in row:
            ch = {"messages": row.get("messages_chosen") or []}
            rj = {"messages": row.get("messages_rejected") or []}
        else:
            prompt = row.get("prompt", "")
            p_msgs = prompt if isinstance(prompt, list) else [
                {"role": "user", "content": str(prompt)}]
            ch = {"messages": p_msgs + self._to_messages(row.get("chosen", ""))[1:]}
            rj = {"messages": p_msgs + self._to_messages(row.get("rejected", ""))[1:]}
        c = self._fmt._process_item(ch)
        r = self._fmt._process_item(rj)
        if c is None or r is None:
            return None
        return {"chosen": c, "rejected": r}

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        count = 0
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if self.max_samples and count >= self.max_samples:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                try:
                    pair = self._row_to_pair(row)
                except Exception:
                    continue
                if pair is not None:
                    count += 1
                    yield pair


def pref_collate_fn(batch, pad_token_id: int = 0) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        "chosen": collate_fn([b["chosen"] for b in batch], pad_token_id),
        "rejected": collate_fn([b["rejected"] for b in batch], pad_token_id),
    }


def _current_line_of(ds) -> int:
    """Номер строки файла, на которой поток датасета стоит сейчас.

    ReplayMixedDataset прячет основной датасет под .main — смотрим туда.
    """
    inner = getattr(ds, "main", ds)
    return int(getattr(inner, "current_line", 0) or 0)


def _block_causal_mask(doc_ids, length: int) -> torch.Tensor:
    """Блочная причинная маска (L, L) для упакованных документов.

    Токен видит только свой документ и только токены слева от себя.
    Между документами — -inf, поэтому диалоги не подглядывают друг в друга.
    Строки паддинга (doc_id = -1) разрешаем смотреть только на себя: иначе
    строка из одних -inf даёт NaN в softmax и NaN в градиентах.
    """
    doc = torch.tensor(doc_ids, dtype=torch.long)
    valid = doc >= 0
    same = doc[:, None] == doc[None, :]
    causal = torch.ones(length, length, dtype=torch.bool).tril()
    allow = same & causal & valid[:, None] & valid[None, :]
    allow = allow | (~valid[:, None] & torch.eye(length, dtype=torch.bool))
    mask = torch.zeros(length, length, dtype=torch.float32)
    mask.masked_fill_(~allow, float("-inf"))
    return mask


class LengthGroupedDataset(IterableDataset):
    """Length grouping: сэмплы сортируются по длине внутри окна.

    Батч, собранный из короткого и длинного диалога, тянет паддинг до самой
    длинной строки. Если в батче сэмплы одной длины — лишних токенов почти
    нет. Сортировка идёт ОКНОМ (по умолчанию 64 сэмпла), а не по всему
    файлу: полная сортировка сломала бы мгновенное продолжение по номеру
    строки (resume_state/start_line).
    """

    def __init__(self, base, window: int = 64):
        super().__init__()
        self.base = base
        self.window = max(1, int(window))
        self._window_start = None

    @property
    def current_line(self) -> int:
        # Пока окно не выдано целиком, точка продолжения — его начало:
        # иначе невыданные сэмплы из буфера потерялись бы при остановке.
        if self._window_start is not None:
            return self._window_start
        return _current_line_of(self.base)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        buf = []
        for sample in self.base:
            if not buf:
                self._window_start = _current_line_of(self.base)
            buf.append(sample)
            if len(buf) >= self.window:
                buf.sort(key=lambda x: x["input_ids"].size(0))
                for item in buf:
                    yield item
                buf = []
        if buf:
            buf.sort(key=lambda x: x["input_ids"].size(0))
            for item in buf:
                yield item


class PackedDataset(IterableDataset):
    """Packing: несколько диалогов склеиваются в одно окно max_length.

    Документы не смешиваются: между ними блочная маска внимания (4D), а
    позиции (position_ids) сбрасываются на каждом документе — диалог в
    упакованном окне выглядит для модели ровно так же, как отдельный.
    Лосс считается только по токенам ассистента, как и без упаковки.

    Выигрыш: в один форвард уходит целое окно токенов. На bs=1 это в
    несколько раз больше полезной работы, чем один диалог за раз.
    """

    def __init__(self, base, max_length: int = 512, pad_token_id: int = 0):
        super().__init__()
        self.base = base
        self.max_length = int(max_length)
        self.pad_token_id = int(pad_token_id)
        self._window_start = None

    @property
    def current_line(self) -> int:
        if self._window_start is not None:
            return self._window_start
        return _current_line_of(self.base)

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        ids_buf: list = []
        labels_buf: list = []
        starts: list = []

        for sample in self.base:
            ids = sample["input_ids"].tolist()
            labels = sample["labels"].tolist()

            if len(ids_buf) + len(ids) > self.max_length:
                # Диалог не влезает в остаток окна — закрываем окно,
                # невыбранное место закроется паддингом.
                yield self._make_window(ids_buf, labels_buf, starts)
                ids_buf, labels_buf, starts = [], [], []

            if not ids_buf:
                # Началось новое окно — запоминаем строку, с которой оно
                # начинается: с неё продолжим после остановки.
                self._window_start = _current_line_of(self.base)

            starts.append(len(ids_buf))
            ids_buf.extend(ids)
            labels_buf.extend(labels)

            if len(ids_buf) >= self.max_length:
                yield self._make_window(ids_buf, labels_buf, starts)
                ids_buf, labels_buf, starts = [], [], []

        if ids_buf:
            yield self._make_window(ids_buf, labels_buf, starts)

    def _make_window(self, ids, labels, starts):
        length = self.max_length
        n = len(ids)
        pad = length - n

        input_ids = ids + [self.pad_token_id] * pad
        attn = [1] * n + [0] * pad

        lab = list(labels)
        # Первый токен документа не должен быть целью: в упакованном окне его
        # предсказывают токены ПРЕДЫДУЩЕГО документа. Без отдельного окна
        # первый токен цели не является (сдвиг лосса), поэтому тут ровно так же.
        for s in starts:
            if s < len(lab):
                lab[s] = -100
        lab += [-100] * pad

        doc_ids = [-1] * length
        position_ids = [0] * length
        for d, s in enumerate(starts):
            e = starts[d + 1] if d + 1 < len(starts) else n
            for i in range(s, e):
                doc_ids[i] = d
                position_ids[i] = i - s

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(lab, dtype=torch.long),
            "position_ids": torch.tensor(position_ids, dtype=torch.long),
            "block_mask": _block_causal_mask(doc_ids, length),
        }
