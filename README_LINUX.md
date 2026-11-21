# Local Forge — Linux beta

## Требования
- Ubuntu 22.04/24.04 или совместимый дистрибутив x86_64
- Python 3.10+ и `python3-venv`
- OpenGL/X11 или Wayland
- Для сборки: Rust stable, `pkg-config`, `libxkbcommon-dev`, `libwayland-dev`, `libx11-dev`, `libxi-dev`, `libgl1-mesa-dev`

## Быстрый старт (готовый архив)
1. Распакуй `local-forge-0.2.0-beta.1-linux-x64.tar.gz`.
2. `./setup.sh` — создаст `.venv` и поставит CPU-зависимости.
3. `./local-forge`
4. Модели скачиваются во вкладке «Хаб»; в архив они не входят.
5. Для Heretic: установи `uv` и запусти `./setup_heretic.sh`.

## Сборка из исходников
```bash
sudo apt install -y build-essential pkg-config libxkbcommon-dev libwayland-dev \
  libx11-dev libxi-dev libgl1-mesa-dev python3-venv
cargo build --release
./setup.sh
./target/release/local-forge
```

## llama_bin
GGUF-чату нужен `llama_bin/llama-server`. Возьми Linux-сборку llama.cpp
(`llama-*-bin-ubuntu-x64.zip`) и распакуй содержимое `build/bin/` в `llama_bin/`,
либо собери llama.cpp сам.

## Ограничения беты
- CPU-сборка; CUDA не входит.
- Публичный API использует ключ `1` без HTTPS — только Radmin VPN/закрытая сеть.
- Модели и датасеты в репозиторий/архив не входят.
