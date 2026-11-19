# Local Forge 0.2.0-beta.1 — source

Локальный чат с GGUF-моделями, хаб Hugging Face, дообучение (SFT / ORPO / KD / Heretic) и публичный OpenAI-совместимый API. Интерфейс RU/EN.

## Сборка из исходников

1. Установи Rust (stable) и Python 3.10+.
2. `setup.bat` — создаст `.venv` и поставит зависимости движка.
3. `cargo build --release` — соберёт `target/release/local-forge.exe`.
4. Для GGUF-чата нужен `llama_bin/llama-server.exe`. Возьми `llama_bin` из release-asset `local-forge-0.2.0-beta.1-win64.zip` или собери llama.cpp и положи бинарники туда.
5. Запусти `local-forge.exe`. Модели качаются во вкладке «Хаб» и в репозиторий не входят.

Heretic (опционально): установи [uv](https://docs.astral.sh/uv/) и запусти `setup_heretic.bat`.

## Что лежит в репозитории
- `src/` — Rust-приложение (egui, M3 UI).
- `python/` — движок чата/обучения.
- `heretic-2.1.0/` — сторонний AGPL-компонент ARA.
- `requirements.txt`, `setup.bat`, `setup_heretic.bat`.
- `colab/`, `distill/` — вспомогательные инструменты дистилляции (без датасетов).

## Лицензия
Проект распространяется под **AGPL-3.0-or-later**, потому что включает Heretic 2.1.0. Если Heretic вынести в отдельную опциональную зависимость, лицензию Local Forge можно сменить (например, на MIT/Apache-2.0).

## Безопасность
Публичный API использует фиксированный ключ `1` и не использует HTTPS. Только Radmin VPN / закрытая сеть, не открывай порт в интернет.

## EN
Local Forge beta source: local GGUF chat, HF hub, SFT/ORPO/KD/Heretic training and an OpenAI-compatible API. Build with `cargo build --release`; run `setup.bat` for Python deps; get `llama_bin` from the release asset or build llama.cpp. AGPL-3.0-or-later because Heretic is included. Public API key `1`, no HTTPS — Radmin VPN only.
