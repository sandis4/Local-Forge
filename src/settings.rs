//! Единые настройки приложения: внешний вид, параметры чата,
//! обучение по умолчанию и пути. Хранятся в %APPDATA%\LocalForge.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub dark: bool,
    /// ru | en — язык интерфейса (см. i18n.rs).
    pub lang: String,
    /// "#rrggbb" или "system" — брать системный акцент.
    pub seed: String,
    pub chat_threads: u32,
    pub chat_ctx: u32,
    pub chat_kv: String,
    pub chat_max_tokens: u32,
    pub chat_temp: f32,
    /// Публичный OpenAI-совместимый API к текущей модели (Radmin VPN).
    pub chat_api: bool,
    pub chat_api_port: u32,
    /// Пусто — автоопределение Radmin VPN; иначе конкретный IP.
    pub chat_api_host: String,
    /// Включённые режимы чата: агент, размышление, веб-поиск.
    pub chat_agent: bool,
    pub chat_think: bool,
    pub chat_web: bool,
    /// duckduckgo | bing | google | yandex
    pub chat_web_engine: String,
    /// Папка проекта для агента; пусто — рабочая папка.
    pub agent_root: String,
    /// Добавленные пользователем проекты и активный из них.
    pub projects: Vec<String>,
    pub active_project: String,
    /// Автор, чьи релизы показывать первыми в хабе.
    pub hf_user: String,
    /// unsloth | runnable — какие кванты собирать в ленту.
    pub hub_quants: String,
    pub train_preset: String,
    /// Пусто — определить автоматически.
    pub python: String,
    pub engine: String,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            dark: false,
            lang: "ru".into(),
            seed: "system".into(),
            chat_threads: 8,
            chat_ctx: 2048,
            chat_kv: "f16".into(),
            chat_max_tokens: 256,
            chat_temp: 0.7,
            chat_api: false,
            chat_api_port: 8000,
            chat_api_host: String::new(),
            chat_agent: false,
            chat_think: false,
            chat_web: false,
            chat_web_engine: "duckduckgo".into(),
            agent_root: String::new(),
            projects: Vec::new(),
            active_project: String::new(),
            hf_user: "YoSaNoo".into(),
            hub_quants: "runnable".into(),
            train_preset: "work".into(),
            python: String::new(),
            engine: String::new(),
        }
    }
}

pub fn path() -> Option<PathBuf> {
    if let Ok(appdata) = std::env::var("APPDATA") {
        let dir = PathBuf::from(appdata).join("LocalForge");
        let _ = std::fs::create_dir_all(&dir);
        return Some(dir.join("settings.json"));
    }
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.join("local-forge.json"))
}

pub fn load() -> AppSettings {
    let Some(p) = path() else {
        return AppSettings::default();
    };
    let Ok(text) = std::fs::read_to_string(p) else {
        return AppSettings::default();
    };
    serde_json::from_str(&text).unwrap_or_default()
}

pub fn save(s: &AppSettings) {
    if let Some(p) = path() {
        if let Ok(text) = serde_json::to_string_pretty(s) {
            let _ = std::fs::write(p, text);
        }
    }
}

pub fn parse_seed(text: &str) -> Option<[u8; 3]> {
    let t = text.trim().trim_start_matches('#');
    if t.len() != 6 {
        return None;
    }
    Some([
        u8::from_str_radix(&t[0..2], 16).ok()?,
        u8::from_str_radix(&t[2..4], 16).ok()?,
        u8::from_str_radix(&t[4..6], 16).ok()?,
    ])
}

pub fn seed_hex(seed: [u8; 3]) -> String {
    format!("#{:02x}{:02x}{:02x}", seed[0], seed[1], seed[2])
}

/// Цвет схемы с учётом режима "system".
pub fn resolve_seed(text: &str) -> [u8; 3] {
    if text.trim() == "system" {
        return crate::sysinfo::accent_color().unwrap_or(crate::color::DEFAULT_SEED);
    }
    parse_seed(text).unwrap_or(crate::color::DEFAULT_SEED)
}
