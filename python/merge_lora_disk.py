"""
Послойное слияние LoRA в базу без загрузки всей модели в RAM.
Читает шарды базы по одному, применяет дельты W += (B@A)*scale, пишет в выход.
Пик RAM: один шард (~5 ГБ) + fp32-копия текущего тензора.
"""

import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = "models/Qwen-3.5-9B"
LORA_DIR = "output_forge1/final_model"
OUT = "models/forge1_merged"

def main():
    ac = json.load(open(os.path.join(LORA_DIR, "adapter_config.json"), encoding="utf-8"))
    r, alpha = ac.get("r", 8), ac.get("lora_alpha", 16)
    scale = alpha / r
    print(f"LoRA: r={r}, alpha={alpha}, scale={scale}")

    # Карта адаптера: "....q_proj.lora_A.weight" -> тензор
    with safe_open(os.path.join(LORA_DIR, "adapter_model.safetensors"), framework="pt", device="cpu") as f:
        lora = {k: f.get_tensor(k) for k in f.keys()}

    pairs = {}  # нормализованный ключ -> (A, B)
    for k, t in lora.items():
        if ".lora_A.weight" in k:
            base_k = k.replace(".lora_A.weight", ".weight").replace("base_model.model.", "")
            pairs.setdefault(base_k, {})[ "A" ] = t.float()
        elif ".lora_B.weight" in k:
            base_k = k.replace(".lora_B.weight", ".weight").replace("base_model.model.", "")
            pairs.setdefault(base_k, {})[ "B" ] = t.float()
    complete = {k: v for k, v in pairs.items() if "A" in v and "B" in v}
    print(f"целей для слияния: {len(complete)}")

    def norm(key: str) -> str:
        # чекпоинт мультимодальный (model.language_model.*), адаптер — от текстового скелета (model.*)
        return key.replace(".language_model.", ".")

    index = json.load(open(os.path.join(BASE, "model.safetensors.index.json"), encoding="utf-8"))["weight_map"]
    shards = sorted(set(index.values()))
    os.makedirs(OUT, exist_ok=True)

    for shard in shards:
        out_shard = os.path.join(OUT, shard)
        if os.path.exists(out_shard):
            os.remove(out_shard)
        tensors = {}
        merged_n = 0
        with safe_open(os.path.join(BASE, shard), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            for key in keys:
                t = f.get_tensor(key)
                target = complete.get(norm(key))
                if target is not None:
                    A, B = target["A"], target["B"]
                    orig_dtype = t.dtype
                    W = t.float()
                    delta = (B @ A) * scale
                    W += delta.to(W.dtype)
                    t = W.to(orig_dtype).contiguous()
                    merged_n += 1
                tensors[key] = t
        save_file(tensors, out_shard, metadata={"format": "pt"})
        print(f"{shard}: {len(tensors)} тензоров, слито {merged_n} | RAM: {__import__('psutil').Process().memory_info().rss/1e9:.1f} ГБ")
        del tensors

    # Копируем конфиги и токенизатор
    for fn in os.listdir(BASE):
        if fn.endswith(".safetensors") or fn == "model.safetensors.index.json":
            continue
        src = os.path.join(BASE, fn)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(OUT, fn))
    # индекс с теми же именами шардов
    shutil.copy2(os.path.join(BASE, "model.safetensors.index.json"), os.path.join(OUT, "model.safetensors.index.json"))
    print(f"Готово: {OUT}")

if __name__ == "__main__":
    main()
