"""
Слияние LoRA-адаптеров (task arithmetic / TIES-lite) — единственный способ
«соединить две модели», который работает без GPU-фермы: адаптеры должны быть
обучены на ОДНОЙ базе (одинаковые формы и токенизатор).

Математика: для каждого модуля ΔW_i = B_i @ A_i * (alpha_i / r_i).
Слитая дельта: Δ = Σ w_i * ΔW_i, затем SVD -> обратно в A/B ранга r.

  python merge_loras.py --base models/Qwen-3.5-9B \
      --adapters output_forge1_1/final_model:1.0 some_code_lora:0.5 \
      --out merged/forge_code --r 16

Почему нельзя «Gemma 4 + маленькая модель»: у них разные словари и формы
тензоров — тензоры просто не совпадут, склеивать нечего.
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file


def load_adapter(path):
    """-> (state_dict, config)"""
    f = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(f):
        raise SystemExit(f"Нет {f}")
    cfg_p = os.path.join(path, "adapter_config.json")
    cfg = json.load(open(cfg_p, encoding="utf-8")) if os.path.exists(cfg_p) else {}
    return load_file(f), cfg


def module_pairs(sd):
    """{module_key: (A, B)} из ключей PEFT."""
    pairs = {}
    for k, v in sd.items():
        if ".lora_A" in k:
            base = k.split(".lora_A")[0]
            pairs.setdefault(base, {})["A"] = v
        elif ".lora_B" in k:
            base = k.split(".lora_B")[0]
            pairs.setdefault(base, {})["B"] = v
    return {k: (v["A"], v["B"]) for k, v in pairs.items() if "A" in v and "B" in v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True,
                    help="пути к LoRA-адаптерам, опционально :вес (по умолчанию 1.0)")
    ap.add_argument("--base", default=None, help="путь к базовой модели (для adapter_config)")
    ap.add_argument("--out", required=True, help="куда сохранить слитый адаптер")
    ap.add_argument("--r", type=int, default=None, help="ранг результата (по умолчанию как у первого)")
    ap.add_argument("--ties", action="store_true", help="TIES: обрезка мелких значений дельты")
    args = ap.parse_args()

    items = []
    for spec in args.adapters:
        # вес отделяем по ПОСЛЕДНЕМУ двоеточию (иначе Windows-путь "C:\..." ломается)
        path, w = spec, 1.0
        if ":" in spec:
            head, tail = spec.rsplit(":", 1)
            try:
                w = float(tail)
                path = head
            except ValueError:
                path, w = spec, 1.0
        sd, cfg = load_adapter(path)
        items.append((path, w, sd, cfg))
        print(f"[{path}] вес {w}, r={cfg.get('r')} alpha={cfg.get('lora_alpha')}")

    # проверка совместимости
    base_sd = items[0][2]
    ref_pairs = module_pairs(base_sd)
    ref_cfg = items[0][3]
    for path, _, sd, _ in items[1:]:
        pairs = module_pairs(sd)
        if set(pairs) != set(ref_pairs):
            missing = len(set(ref_pairs) - set(pairs))
            extra = len(set(pairs) - set(ref_pairs))
            raise SystemExit(f"Адаптер '{path}' несовместим: модулей не хватает {missing}, лишних {extra}. "
                             f"Они обучены на разных базах/архитектурах — слияние невозможно.")
        for k in ref_pairs:
            if pairs[k][0].shape != ref_pairs[k][0].shape or pairs[k][1].shape != ref_pairs[k][1].shape:
                raise SystemExit(f"Формы различаются в {k}: {pairs[k][0].shape} vs {ref_pairs[k][0].shape}")

    r_out = args.r or int(ref_cfg.get("r", 8))
    print(f"Модулей: {len(ref_pairs)} | ранг результата: {r_out}")

    out_sd = {}
    for key, (A0, B0) in ref_pairs.items():
        delta = None
        for path, w, sd, cfg in items:
            A, B = module_pairs(sd)[key]
            scale = float(cfg.get("lora_alpha", 1)) / float(cfg.get("r", 1))
            d = (B.float() @ A.float()) * scale * w
            delta = d if delta is None else delta + d
        if args.ties:
            flat = delta.reshape(-1)
            thr = flat.abs().quantile(0.8)
            delta = torch.where(delta.abs() < thr, torch.zeros_like(delta), delta)
        # обратно в A/B нужного ранга
        U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
        r = min(r_out, S.numel())
        U, S, Vh = U[:, :r], S[:r], Vh[:r]
        root = S.sqrt()
        B_new = (U * root.unsqueeze(0)).to(torch.float32)
        A_new = (root.unsqueeze(1) * Vh).to(torch.float32)
        # имена в стиле PEFT: тот же префикс, что у первого адаптера
        for k in base_sd:
            if k.startswith(key + ".lora_A"):
                a_name = k
            if k.startswith(key + ".lora_B"):
                b_name = k
        out_sd[a_name] = A_new.contiguous()
        out_sd[b_name] = B_new.contiguous()

    os.makedirs(args.out, exist_ok=True)
    save_file(out_sd, os.path.join(args.out, "adapter_model.safetensors"),
              metadata={"format": "pt"})

    cfg_out = dict(ref_cfg)
    cfg_out.update({"r": r_out, "lora_alpha": r_out,   # scale = 1: дельта уже точная
                    "peft_type": "LORA", "task_type": "CAUSAL_LM"})
    if args.base:
        cfg_out["base_model_name_or_path"] = args.base
    json.dump(cfg_out, open(os.path.join(args.out, "adapter_config.json"), "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)
    # токенизатор — от первой базы, если он есть рядом с адаптером
    for fn in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = os.path.join(os.path.dirname(items[0][0]), fn)
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, os.path.join(args.out, fn))
    print(f"Готово: {args.out} ({len(out_sd)} тензоров, r={r_out})")


if __name__ == "__main__":
    main()
