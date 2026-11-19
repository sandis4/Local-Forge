"""Быстрая диагностика состояния обучения (без загрузки модели).

  python state_info.py output_little_11

Показывает: шаг, строка файла, число групп оптимизатора, совпадают ли
веса с моментом сохранения, и что произойдёт при продолжении.
"""

import json
import os
import sys

import torch


def info(out_dir):
    p = os.path.join(out_dir, "training_state", "training_state.pt")
    print(f"=== {out_dir} ===")
    # показываем и основной, и резервный
    for label, cand in (("основной", p), ("резервный", p + ".prev")):
        if not os.path.exists(cand):
            print(f"  {label}: нет")
            continue
        st = os.stat(cand)
        try:
            d = torch.load(cand, map_location="cpu", weights_only=False)
            print(f"  {label}: шаг {d.get('global_step')} | строка {d.get('line')} | "
                  f"эпоха {d.get('epoch')} | {st.st_size/1e6:.2f} МБ "
                  f"({__import__('datetime').datetime.fromtimestamp(st.st_mtime):%d.%m.%Y %H:%M})")
            if cand == p:
                main = d
        except Exception as e:
            print(f"  {label}: БИТЫЙ ({str(e)[:60]}) — {st.st_size/1e6:.2f} МБ")
    if not os.path.exists(p) and not os.path.exists(p + ".prev"):
        print("  состояния нет — продолжение начнётся с шага 1")
        print("  (восстановить можно так: --resume_step auto --resume "
              f"{os.path.join(out_dir, 'interactive_checkpoint')})")
        return
    try:
        d = torch.load(p, map_location="cpu", weights_only=False)
    except Exception:
        print("  ВЕРДИКТ: основной битый — возьмётся резервный (.prev), если он цел")
        return
    print(f"  оптимизатор: {d.get('optimizer_name')} | групп: "
          f"{len((d.get('optimizer') or {}).get('param_groups', []))} | "
          f"состояний: {len((d.get('optimizer') or {}).get('state', {}))}")
    print(f"  аккумуляция: {d.get('gradient_accumulation_steps')} | LR: {d.get('lr')} | "
          f"LISA: {d.get('lisa_active')}")
    sig = d.get("weights_sig")
    wf = os.path.join(out_dir, "interactive_checkpoint", "adapter_model.safetensors")
    cur = None
    if os.path.exists(wf):
        s = os.stat(wf)
        cur = [os.path.basename(wf), s.st_mtime_ns, s.st_size]
    print(f"  веса совпадают: {'ДА — восстановится точно' if sig == cur else 'НЕТ — шаг сохранится, моменты Adam заново'}")


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["output_little_11", "output_forge1_1"]
    for d in dirs:
        if os.path.isdir(d):
            info(d)
            print()
