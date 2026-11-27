//! Чат: GGUF через llama.cpp (быстро) или transformers + LoRA.
//! Движок — python/chat_engine.py, общение JSON-строками.

use crate::i18n::tr;
use std::collections::{HashMap, VecDeque};
use std::path::Path;

use eframe::egui::{self, RichText, Rounding, ScrollArea};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::models;
use crate::runner::Runner;
use crate::theme::{self, ButtonKind, Theme};

#[derive(Clone)]
pub struct Msg {
    pub role: String,
    pub content: String,
}

#[derive(Clone)]
pub struct ModelEntry {
    pub path: String,
    pub size_mb: f64,
    /// Полное число параметров из весов (GGUF-заголовок / safetensors).
    pub total: Option<u64>,
    /// «E2B» в имени = активные параметры, а не все.
    pub active: Option<String>,
}

#[derive(PartialEq, Clone, Copy)]
pub enum Status {
    Unloaded,
    Loading,
    Ready,
    Generating,
    Failed,
}

#[derive(PartialEq, Clone, Copy)]
pub enum ChatAction {
    OpenHub,
}

fn engine_label(id: &str) -> &'static str {
    match id {
        "bing" => "Bing",
        "google" => "Google",
        "yandex" => tr("Яндекс"),
        _ => "DuckDuckGo",
    }
}

fn entry_label(m: &ModelEntry) -> String {
    let mut line = m.path.clone();
    if let Some(total) = m.total {
        line.push_str(&format!("   · {}", models::fmt_params(total)));
    }
    if let Some(act) = &m.active {
        line.push_str(&format!(" ({act})"));
    }
    line.push_str(&crate::tf!("   · {:.0} МБ", "   · {:.0} MB", m.size_mb));
    line
}

/// «E2B» в имени — это эффективные (активные) параметры, а не все.
fn active_hint(name: &str) -> Option<String> {
    let upper = name.to_uppercase();
    let bytes = upper.as_bytes();
    for i in 1..bytes.len() {
        if bytes[i] != b'B' || !bytes[i - 1].is_ascii_digit() {
            continue;
        }
        let mut j = i;
        while j > 0 && (bytes[j - 1].is_ascii_digit() || bytes[j - 1] == b'.') {
            j -= 1;
        }
        if j > 0 && bytes[j - 1] == b'E' {
            return Some(crate::tf!("{} акт.", "{} active", &upper[j..=i]));
        }
    }
    None
}

fn dir_size_mb(path: &Path) -> f64 {
    let mut total = 0u64;
    if let Ok(entries) = std::fs::read_dir(path) {
        for e in entries.flatten() {
            if let Ok(m) = e.metadata() {
                if m.is_file() {
                    total += m.len();
                }
            }
        }
    }
    total as f64 / (1024.0 * 1024.0)
}

pub struct Chat {
    pub model: String,
    pub adapter: String,
    pub threads: String,
    pub ctx: String,
    pub max_tokens: String,
    pub temp: String,
    pub kv_type: String,
    pub messages: Vec<Msg>,
    pub input: String,
    pub models: Vec<ModelEntry>,
    pub adapters: Vec<String>,
    pub status: Status,
    pub backend: String,
    pub info: String,
    pub tok_s: Option<f64>,
    pub loading_msg: String,
    pub error: Option<String>,
    pub log: VecDeque<String>,
    pub agent_root: String,
    pub agent_steps: Vec<(String, String)>,
    pub agent_answer: String,
    pub agent_meta: String,
    pub agent_running: bool,
    /// Дополнения к чату: агент (файлы), размышление, веб-поиск.
    pub agent: bool,
    pub think: bool,
    pub web: bool,
    pub api_enabled: bool,
    pub api_port: String,
    pub api_host: String,
    pub api_running: bool,
    pub api_url: String,
    pub api_key: String,
    pub api_note: String,
    pub web_engine: String,
    pub reasoning: String,
    pub web_note: String,
    show_adv: bool,
    workspace: String,
    auto_load: bool,
    scan_rx: Option<std::sync::mpsc::Receiver<(Vec<ModelEntry>, Vec<String>)>>,
    params_rx: Option<std::sync::mpsc::Receiver<Vec<(String, Option<u64>)>>>,
    engine: Option<Runner>,
}

fn split_shard(name: &str) -> Option<(String, u32, u32)> {
    let lower = name.to_lowercase();
    let stem = lower.strip_suffix(".gguf")?;
    let (left, right) = stem.rsplit_once("-of-")?;
    let total: u32 = right.parse().ok()?;
    let (base, part) = left.rsplit_once('-')?;
    let part: u32 = part.parse().ok()?;
    if total < 2 || part == 0 || part > total {
        return None;
    }
    Some((base.to_string(), part, total))
}

fn rank(m: &ModelEntry) -> u8 {
    let u = m.path.to_uppercase();
    if !u.ends_with(".GGUF") {
        5
    } else if u.contains("Q4") {
        0
    } else if u.contains("Q5") || u.contains("Q6") {
        1
    } else if u.contains("Q8") {
        2
    } else if u.contains("Q2") || u.contains("Q3") {
        3
    } else {
        4
    }
}

/// HF-модели из `models/` + GGUF из `models_gguf/`, отсортированные (rank, path).
fn scan_models(ws: &Path) -> Vec<ModelEntry> {
    let mut models: Vec<ModelEntry> = Vec::new();
    if let Ok(dirs) = std::fs::read_dir(ws.join("models")) {
        for d in dirs.flatten() {
            let p = d.path();
            if p.join("config.json").exists() {
                let name = d.file_name().to_string_lossy().to_string();
                models.push(ModelEntry {
                    path: format!("models/{name}"),
                    size_mb: dir_size_mb(&p),
                    total: None,
                    active: active_hint(&name),
                });
            }
        }
    }
    models.extend(scan_splits(ws));
    models.sort_by_key(|m| (rank(m), m.path.clone()));
    models
}

/// GGUF может быть разбит на шарды: list-*.gguf показывает только первый,
/// llama.cpp сам подхватит остальные из той же папки.
fn scan_splits(ws: &Path) -> Vec<ModelEntry> {
    let mut models = Vec::new();
    let Ok(files) = std::fs::read_dir(ws.join("models_gguf")) else {
        return models;
    };
    let mut splits: HashMap<String, (String, String, f64, u32, u32)> = HashMap::new();
    for f in files.flatten() {
        let p = f.path();
        if !p.extension().map(|e| e.eq_ignore_ascii_case("gguf")).unwrap_or(false) {
            continue;
        }
        let name = f.file_name().to_string_lossy().to_string();
        let size = f.metadata().map(|m| m.len()).unwrap_or(0) as f64 / (1024.0 * 1024.0);
        if let Some((base, part, total)) = split_shard(&name) {
            let e = splits.entry(base).or_insert_with(|| (String::new(), String::new(), 0.0, 0, total));
            e.2 += size;
            e.3 = e.3.max(part);
            e.4 = e.4.max(total);
            if part == 1 {
                e.0 = format!("models_gguf/{name}");
                e.1 = name.clone();
            }
        } else {
            models.push(ModelEntry {
                path: format!("models_gguf/{name}"),
                size_mb: size,
                total: None,
                active: active_hint(&name),
            });
        }
    }
    for (_, (path, name, size, _seen, _total)) in splits {
        if path.is_empty() {
            continue;
        }
        models.push(ModelEntry {
            path,
            size_mb: size,
            total: None,
            active: active_hint(&name),
        });
    }
    models
}

/// Папки `output*/` с `adapter_config.json`; первый элемент — «без адаптера».
fn scan_adapters(ws: &Path) -> Vec<String> {
    let mut adapters = vec![String::new()];
    if let Ok(dirs) = std::fs::read_dir(ws) {
        for d in dirs.flatten() {
            let p = d.path();
            let name = d.file_name().to_string_lossy().to_string();
            if !p.is_dir() || !name.starts_with("output") {
                continue;
            }
            if let Ok(subs) = std::fs::read_dir(&p) {
                for s in subs.flatten() {
                    if s.path().join("adapter_config.json").exists() {
                        adapters.push(format!("{}/{}", name, s.file_name().to_string_lossy()));
                    }
                }
            }
        }
    }
    adapters
}

fn scan_workspace(ws: &Path) -> (Vec<ModelEntry>, Vec<String>) {
    (scan_models(ws), scan_adapters(ws))
}

/// События chat_engine.py: `{"ev": "...", ...}`.
#[derive(Deserialize)]
#[serde(tag = "ev")]
enum ChatEvent {
    #[serde(rename = "loading")]
    Loading {
        #[serde(default)]
        message: String,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        step: u64,
    },
    #[serde(rename = "heartbeat")]
    Heartbeat,
    #[serde(rename = "tool")]
    Tool {
        #[serde(default)]
        name: String,
        #[serde(default)]
        args: Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        #[serde(default)]
        ok: bool,
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "reasoning")]
    Reasoning {
        #[serde(default)]
        text: Option<String>,
    },
    #[serde(rename = "assistant")]
    Assistant {
        #[serde(default)]
        text: String,
    },
    #[serde(rename = "agent_done")]
    AgentDone,
    #[serde(rename = "web")]
    Web {
        #[serde(default)]
        engine: String,
        #[serde(default)]
        found: u64,
        #[serde(default)]
        note: String,
    },
    #[serde(rename = "ready")]
    Ready {
        #[serde(default)]
        backend: String,
        #[serde(default)]
        info: String,
    },
    #[serde(rename = "api")]
    Api {
        #[serde(default)]
        running: bool,
        #[serde(default)]
        url: String,
        #[serde(default)]
        key: Option<String>,
        #[serde(default)]
        note: String,
        #[serde(default)]
        error: Option<String>,
    },
    #[serde(rename = "token")]
    Token {
        #[serde(default)]
        text: Option<String>,
    },
    #[serde(rename = "done")]
    Done {
        #[serde(default)]
        tok_s: Option<f64>,
    },
    #[serde(rename = "error")]
    Error {
        #[serde(default)]
        message: Option<String>,
    },
}

impl Chat {
    pub fn new(ws: &Path) -> Self {
        let mut chat = Self {
            model: String::new(),
            adapter: String::new(),
            threads: "8".into(),
            ctx: "2048".into(),
            max_tokens: "256".into(),
            temp: "0.7".into(),
            kv_type: "f16".into(),
            messages: Vec::new(),
            input: String::new(),
            models: Vec::new(),
            adapters: Vec::new(),
            status: Status::Unloaded,
            backend: String::new(),
            info: String::new(),
            tok_s: None,
            loading_msg: String::new(),
            error: None,
            log: VecDeque::new(),
            agent_root: String::new(),
            agent_steps: Vec::new(),
            agent_answer: String::new(),
            agent_meta: String::new(),
            agent_running: false,
            agent: false,
            think: false,
            web: false,
            api_enabled: false,
            api_port: "8000".into(),
            api_host: String::new(),
            api_running: false,
            api_url: String::new(),
            api_key: "1".into(),
            api_note: String::new(),
            web_engine: "duckduckgo".into(),
            reasoning: String::new(),
            web_note: String::new(),
            show_adv: false,
            workspace: ws.display().to_string(),
            auto_load: false,
            scan_rx: None,
            params_rx: None,
            engine: None,
        };
        chat.rescan(ws);
        chat
    }

    pub fn rescan(&mut self, ws: &Path) {
        if self.scan_rx.is_some() {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let ws = ws.to_path_buf();
        std::thread::spawn(move || {
            let _ = tx.send(scan_workspace(&ws));
        });
        self.scan_rx = Some(rx);
    }

    /// true — скан завершился, список обновился.
    pub fn poll_scan(&mut self, ws: &Path) -> bool {
        let Some(rx) = &self.scan_rx else { return false };
        match rx.try_recv() {
            Ok((models, adapters)) => {
                self.models = models;
                self.adapters = adapters;
                if self.model.is_empty() {
                    self.model = self.models.first().map(|m| m.path.clone()).unwrap_or_default();
                }
                self.scan_rx = None;
                self.start_params_probe(ws);
                true
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.scan_rx = None;
                true
            }
            Err(_) => false,
        }
    }

    /// Параметры считаются в фоне: заголовки GGUF с токенизатором
    /// парсятся секунду-две, окно к этому моменту уже открыто.
    fn start_params_probe(&mut self, ws: &Path) {
        if self.params_rx.is_some() {
            return;
        }
        let pending: Vec<String> = self
            .models
            .iter()
            .filter(|m| m.total.is_none())
            .map(|m| m.path.clone())
            .collect();
        if pending.is_empty() {
            return;
        }
        let ws = ws.to_path_buf();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut out = Vec::new();
            for rel in pending {
                let total = models::total_params(&ws.join(&rel));
                out.push((rel, total));
            }
            let _ = tx.send(out);
        });
        self.params_rx = Some(rx);
    }

    pub fn poll_params(&mut self) {
        let Some(rx) = &self.params_rx else { return };
        if let Ok(list) = rx.try_recv() {
            for (path, total) in list {
                if let Some(m) = self.models.iter_mut().find(|m| m.path == path) {
                    m.total = total;
                }
            }
            self.params_rx = None;
        }
    }

    pub fn is_scanning(&self) -> bool {
        self.scan_rx.is_some()
    }


    fn ensure_engine(&mut self, python: &str, engine_dir: &Path, ws: &Path) -> Result<(), String> {
        if self.engine.is_some() {
            return Ok(());
        }
        let script = engine_dir.join("chat_engine.py");
        if !script.exists() {
            return Err(crate::tf!("нет '{}'", "missing '{}'", script.display()));
        }
        let runner = Runner::spawn(python, &[script.to_string_lossy().into_owned()], ws)
            .map_err(|e| crate::tf!("не удалось запустить движок чата: {e}", "failed to start the chat engine: {e}"))?;
        self.engine = Some(runner);
        Ok(())
    }

    fn send_json(&mut self, value: Value) {
        if let Some(r) = self.engine.as_mut() {
            let _ = r.send(&value.to_string());
        }
    }

    pub fn load(&mut self, python: &str, engine_dir: &Path, ws: &Path) {
        self.error = None;
        if let Err(e) = self.ensure_engine(python, engine_dir, ws) {
            self.error = Some(e);
            self.status = Status::Failed;
            return;
        }
        self.messages.clear();
        self.tok_s = None;
        self.status = Status::Loading;
        self.loading_msg = tr("Запускаю движок…").into();
        let threads: u64 = self.threads.trim().parse().unwrap_or(8);
        let ctx: u64 = self.ctx.trim().parse().unwrap_or(2048);
        let api_port: u64 = self.api_port.trim().parse().unwrap_or(8000);
        self.send_json(json!({
            "cmd": "load",
            "model": self.model,
            "adapter": self.adapter,
            "threads": threads,
            "ctx": ctx,
            "kv_type": self.kv_type,
            "api": {"enabled": self.api_enabled, "port": api_port,
                    "host": if self.api_host.trim().is_empty() { "auto" } else { self.api_host.trim() }},
        }));
    }

    pub fn api_apply(&mut self) {
        if self.engine.is_none() {
            return;
        }
        let port: u64 = self.api_port.trim().parse().unwrap_or(8000);
        self.send_json(json!({
            "cmd": "api",
            "enabled": self.api_enabled,
            "port": port,
            "host": if self.api_host.trim().is_empty() { "auto" } else { self.api_host.trim() },
        }));
    }

    pub fn ask(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.status != Status::Ready || self.agent_running {
            return;
        }
        self.input.clear();
        self.messages.retain(|m| m.role == "user" || m.role == "assistant");
        self.messages.push(Msg { role: "user".into(), content: text.clone() });
        self.reasoning.clear();
        self.web_note.clear();

        if self.agent {
            self.agent_steps.clear();
            self.agent_answer.clear();
            self.agent_meta = tr("думаю…").into();
            self.agent_running = true;
            let root = if self.agent_root.trim().is_empty() {
                self.workspace.clone()
            } else {
                self.agent_root.clone()
            };
            let max_tokens: u64 = self.max_tokens.trim().parse().unwrap_or(2048).max(1024);
            self.send_json(json!({
                "cmd": "agent",
                "task": text,
                "root": root,
                "max_steps": 30,
                "max_tokens": max_tokens,
                "temp": 0.3,
                "think": self.think,
                "web": self.web,
                "web_engine": self.web_engine,
            }));
            return;
        }

        self.messages.push(Msg { role: "assistant".into(), content: String::new() });
        let history: Vec<Value> = self
            .messages
            .iter()
            .filter(|m| {
                (m.role == "user" || m.role == "assistant")
                    && !(m.role == "assistant" && m.content.is_empty())
            })
            .map(|m| json!({"role": m.role, "content": m.content}))
            .collect();
        let max_tokens: u64 = self.max_tokens.trim().parse().unwrap_or(256);
        let temp: f64 = self.temp.trim().parse().unwrap_or(0.7);
        self.send_json(json!({
            "cmd": "chat",
            "messages": history,
            "max_tokens": max_tokens,
            "temp": temp,
            "top_p": 0.95,
            "think": self.think,
            "web": self.web,
            "web_engine": self.web_engine,
        }));
        self.status = Status::Generating;
    }

    pub fn stop_agent(&mut self) {
        if self.agent_running {
            self.send_json(json!({"cmd": "stop"}));
            self.agent_meta = tr("останавливаю…").into();
        }
    }

    pub fn stop(&mut self) {
        if self.status == Status::Generating {
            self.send_json(json!({"cmd": "stop"}));
        }
    }

    pub fn new_chat(&mut self) {
        self.messages.clear();
        self.input.clear();
        self.error = None;
        self.tok_s = None;
        self.agent_steps.clear();
        self.agent_answer.clear();
        self.reasoning.clear();
        self.web_note.clear();
    }

    pub fn poll(&mut self) {
        let lines: Vec<String> = match self.engine.as_mut() {
            Some(r) => r.drain(),
            None => return,
        };
        for line in lines {
            self.handle_event(&line);
        }
    }

    /// Разбор одной строки chat_engine.py. Не-JSON и неизвестные события — в журнал.
    fn handle_event(&mut self, line: &str) {
        let event = match serde_json::from_str::<ChatEvent>(line) {
            Ok(event) => event,
            Err(_) => {
                self.log.push_back(line.to_string());
                return;
            }
        };
        match event {
            ChatEvent::Loading { message } => {
                self.loading_msg = message;
            }
            ChatEvent::Thinking { step } => {
                self.agent_meta = crate::tf!("шаг {step}: генерирую аргументы…", "step {step}: generating arguments…");
            }
            ChatEvent::Heartbeat => {
                self.agent_meta = tr("генерирую… долгая запись — это нормально").into();
            }
            ChatEvent::Tool { name, args } => {
                let line = format!("→ {name} {}", args);
                self.agent_steps.push(("tool".into(), line.clone()));
                Self::push_step_msg(&mut self.messages, line);
            }
            ChatEvent::ToolResult { ok, text } => {
                let line = format!("← {} {text}", if ok { "ok" } else { tr("ошибка") });
                self.agent_steps.push(("result".into(), line.clone()));
                Self::push_step_msg(&mut self.messages, line);
            }
            ChatEvent::Reasoning { text } => {
                // Показываем размышления только в режиме Think.
                if self.think {
                    if let Some(t) = text {
                        self.reasoning.push_str(&t);
                        match self.messages.last_mut() {
                            Some(m) if m.role == "reasoning" => m.content.push_str(&t),
                            _ => self.messages.push(Msg {
                                role: "reasoning".into(),
                                content: t,
                            }),
                        }
                    }
                }
            }
            ChatEvent::Assistant { text } => {
                self.agent_answer = text.clone();
                self.messages.push(Msg { role: "assistant".into(), content: text });
            }
            ChatEvent::AgentDone => {
                self.agent_running = false;
                self.agent_meta = tr("готово").into();
            }
            ChatEvent::Web { engine, found, note } => {
                if found == 0 {
                    self.web_note = crate::tf!("Ищу в {engine}…", "Searching {engine}…");
                } else {
                    self.web_note = crate::tf!("{engine}: {found} источников", "{engine}: {found} sources");
                }
                if !note.is_empty() && found > 0 {
                    self.web_note = crate::tf!("{engine}: {found} источников · {note}", "{engine}: {found} sources · {note}");
                }
            }
            ChatEvent::Ready { backend, info } => {
                self.status = Status::Ready;
                self.backend = backend;
                self.info = info;
                self.loading_msg.clear();
            }
            ChatEvent::Api { running, url, key, note, error } => {
                self.api_running = running;
                self.api_url = url;
                if let Some(k) = key {
                    self.api_key = k;
                }
                self.api_note = note;
                if let Some(e) = error {
                    if !e.is_empty() {
                        self.api_note = format!("{e}");
                    }
                }
            }
            ChatEvent::Token { text } => {
                if let Some(t) = text {
                    if let Some(last) = self.messages.last_mut() {
                        last.content.push_str(&t);
                    }
                }
            }
            ChatEvent::Done { tok_s } => {
                self.status = Status::Ready;
                self.tok_s = tok_s;
            }
            ChatEvent::Error { message } => {
                let message = message.unwrap_or_else(|| tr("ошибка").to_string());
                if self.agent_running {
                    self.agent_running = false;
                    self.agent_meta = message.clone();
                }
                self.error = Some(message);
                if self.status == Status::Generating {
                    self.status = Status::Ready;
                } else {
                    self.status = Status::Failed;
                }
            }
        }
    }

    pub fn is_busy(&self) -> bool {
        self.agent_running || matches!(self.status, Status::Loading | Status::Generating)
    }

    fn model_menu(&mut self, ui: &mut egui::Ui, theme: &Theme) -> Option<ChatAction> {
        let mut action = None;
        let models = self.models.clone();
        let current = models
            .iter()
            .find(|m| m.path == self.model)
            .map(|m| m.path.rsplit('/').next().unwrap_or(&m.path).to_string())
            .unwrap_or_else(|| tr("Выбери модель").into());
        ui.scope(|ui| {
            // «Выбери модель» — плоский текст, без серой плашки
            let v = &mut ui.visuals_mut().widgets;
            v.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
            v.inactive.bg_fill = egui::Color32::TRANSPARENT;
            v.hovered.weak_bg_fill = theme.pal.surface_container_high;
            let resp = ui.menu_button(
            RichText::new(current).font(theme::title_font(16.0)),
            |ui| {
                ui.set_min_width(400.0);
                ui.label(theme.label(tr("Локальные модели")));
                if models.is_empty() {
                    ui.label(theme.label(tr("Пока пусто — скачай модель в хабе.")));
                }
                for m in &models {
                    let line = entry_label(m);
                    if ui.selectable_label(self.model == m.path, line).clicked() {
                        self.model = m.path.clone();
                        self.new_chat();
                        self.auto_load = true;
                        ui.close_menu();
                    }
                }
                ui.separator();
                if ui.button(tr("Хаб моделей…")).clicked() {
                    action = Some(ChatAction::OpenHub);
                    ui.close_menu();
                }
            },
        );
            // Треугольник меню рисуем сами: в Roboto нет «▼».
            theme::draw_icon(
                ui.painter(),
                egui::pos2(resp.response.rect.right() + 10.0, resp.response.rect.center().y),
                theme::icons::EXPAND_MORE,
                17.0,
                theme.pal.on_surface_variant,
            );
        });
        action
    }

    pub fn draw(
        &mut self,
        ui: &mut egui::Ui,
        theme: &Theme,
        python: &str,
        engine_dir: &Path,
        ws: &Path,
    ) -> Option<ChatAction> {
        let p = theme.pal;
        let busy = self.is_busy();
        let mut action = None;

        ui.horizontal(|ui| {
            action = self.model_menu(ui, theme);
            if self.is_scanning() {
                ui.add(egui::Spinner::new().size(14.0).color(p.primary));
            }
            if self.status == Status::Loading {
                ui.add(egui::Spinner::new().size(15.0).color(p.primary));
                ui.label(theme.label(&self.loading_msg));
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if theme::button(ui, theme, tr("Обновить"), ButtonKind::Text, !busy).clicked() {
                    self.rescan(ws);
                }
                if let (Some(tok), Status::Ready) = (self.tok_s, self.status) {
                    ui.label(theme.label(crate::tf!("{tok:.2} ток/с", "{tok:.2} tok/s")));
                }
                if !self.backend.is_empty() {
                    theme::chip(ui, theme, &self.backend);
                }
                if theme::button(ui, theme, tr("Параметры"), ButtonKind::Text, true).clicked() {
                    self.show_adv = !self.show_adv;
                }
            });
        });
        if self.auto_load && self.status != Status::Loading {
            self.auto_load = false;
            self.load(python, engine_dir, ws);
        }
        ui.add_space(6.0);

        if let Some(err) = &self.error {
            ui.label(RichText::new(err).color(p.error).size(12.0));
        }

        if self.messages.is_empty() {
            let free = (ui.available_height() - 220.0).max(0.0);
            ui.add_space((free * 0.16).min(100.0));
            let hint = match self.status {
                Status::Unloaded => match self.models.iter().find(|m| m.path == self.model) {
                    Some(m) => {
                        let params = m
                            .active
                            .clone()
                            .or_else(|| m.total.map(models::fmt_params))
                            .unwrap_or_default();
                        if params.is_empty() {
                            crate::tf!("{:.0} МБ", "{:.0} MB", m.size_mb)
                        } else {
                            crate::tf!("{params} · {:.0} МБ", "{params} · {:.0} MB", m.size_mb)
                        }
                    }
                    None => tr("Выбери модель сверху — она загрузится сама.").to_string(),
                },
                Status::Loading => crate::tf!("Загружаю… {}", "Loading… {}", self.loading_msg),
                Status::Ready => format!("{} · {}", self.model, self.info),
                Status::Generating => tr("Генерирую…").to_string(),
                Status::Failed => tr("Не удалось загрузить модель — смотри ошибку выше.").to_string(),
            };
            // Главный типографический ход: крупный левый блок,
            // одна акцентная строка, волосяная линия и моно-метаданные.
            ui.horizontal(|ui| {
                ui.add_space(28.0);
                ui.vertical(|ui| {
                    ui.label(theme.overline(tr("Локальный инференс / CPU")));
                    ui.add_space(10.0);
                    ui.label(
                        RichText::new("Локальный
интеллект.")
                            .font(egui::FontId::new(54.0, egui::FontFamily::Proportional))
                            .color(p.on_surface),
                    );
                    ui.label(
                        RichText::new(tr("Без облака."))
                            .font(egui::FontId::new(54.0, egui::FontFamily::Proportional))
                            .color(p.primary),
                    );
                    ui.add_space(14.0);
                    let (line, _) = ui.allocate_exact_size(
                        egui::vec2(420.0, 1.0),
                        egui::Sense::hover(),
                    );
                    ui.painter().line_segment(
                        [line.left_center(), line.right_center()],
                        egui::Stroke::new(1.0_f32, p.outline_variant),
                    );
                    ui.add_space(10.0);
                    ui.label(
                        theme.mono(
                            if self.model.is_empty() {
                                tr("Модель не выбрана")
                            } else {
                                self.model.as_str()
                            },
                            11.0,
                        )
                        .color(p.on_surface),
                    );
                    ui.add_space(2.0);
                    ui.label(theme.mono(&hint, 10.5));
                });
            });
            // Композер у нижнего края.
            let rest = (ui.available_height() - 166.0).max(0.0);
            ui.add_space(rest);
        } else {
            let h = (ui.available_height() - 190.0).max(160.0);
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), h),
                egui::Layout::top_down(egui::Align::Min),
                |ui| self.messages_panel(ui, theme),
            );
            ui.add_space(8.0);
        }

        self.tools_row(ui, theme);
        ui.add_space(4.0);
        self.composer(ui, theme, python, engine_dir, ws);
        self.advanced_window(ui.ctx(), theme, python, engine_dir, ws, busy);
        action
    }

    fn messages_panel(&mut self, ui: &mut egui::Ui, theme: &Theme) {
        ScrollArea::vertical()
            .auto_shrink([false, false])
            .stick_to_bottom(true)
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                for m in &self.messages {
                    if m.role == "assistant" && m.content.is_empty() {
                        continue;
                    }
                    match m.role.as_str() {
                        "tools" => Self::tools_block(ui, theme, &m.content),
                        "reasoning" => Self::reasoning_block(ui, theme, &m.content),
                        _ => Self::bubble(ui, theme, &m.role, &m.content),
                    }
                }
            });
    }

    fn push_step_msg(messages: &mut Vec<Msg>, line: String) {
        match messages.last_mut() {
            Some(m) if m.role == "tools" => {
                m.content.push('\n');
                m.content.push_str(&line);
            }
            _ => messages.push(Msg { role: "tools".into(), content: line }),
        }
    }

    /// Ход работы агента: компактный моно-блок в ленте чата.
    fn tools_block(ui: &mut egui::Ui, theme: &Theme, text: &str) {
        let p = theme.pal;
        ui.add_space(4.0);
        egui::Frame::none()
            .fill(p.surface_container_low)
            .rounding(Rounding::same(12.0))
            .inner_margin(egui::Margin::symmetric(14.0, 10.0))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(
                    RichText::new(tr("Ход работы"))
                        .font(theme::title_font(12.0))
                        .color(p.on_surface_variant),
                );
                ui.add_space(2.0);
                for line in text.lines() {
                    let color = if line.starts_with('→') {
                        p.primary
                    } else {
                        p.on_surface_variant
                    };
                    ui.label(
                        RichText::new(line)
                            .monospace()
                            .size(11.5)
                            .color(color),
                    );
                }
            });
    }

    fn reasoning_block(ui: &mut egui::Ui, theme: &Theme, text: &str) {
        let p = theme.pal;
        egui::CollapsingHeader::new(
            RichText::new(tr("Размышления"))
                .font(theme::title_font(12.0))
                .color(p.on_surface_variant),
        )
        .id_salt("reasoning")
        .default_open(true)
        .show(ui, |ui| {
            ui.label(
                RichText::new(text)
                    .size(13.0)
                    .italics()
                    .color(p.on_surface_variant),
            );
        });
    }

    /// Пузырь чата: у пользователя скруглён правый нижний угол,
    /// у ответа — левый нижний (как в M3).
    fn bubble(ui: &mut egui::Ui, theme: &Theme, role: &str, text: &str) {
        let p = theme.pal;
        let user = role == "user";
        let (fill, fg) = if user {
            (p.primary_container, p.on_primary_container)
        } else {
            (p.surface_container_high, p.on_surface)
        };
        let rounding = if user {
            egui::Rounding { nw: 20.0, ne: 20.0, sw: 20.0, se: 4.0 }
        } else {
            egui::Rounding { nw: 20.0, ne: 20.0, sw: 4.0, se: 20.0 }
        };
        ui.add_space(4.0);
        let layout = if user {
            egui::Layout::right_to_left(egui::Align::Min)
        } else {
            egui::Layout::left_to_right(egui::Align::Min)
        };
        ui.with_layout(layout, |ui| {
            egui::Frame::none()
                .fill(fill)
                .rounding(rounding)
                .inner_margin(egui::Margin::symmetric(16.0, 12.0))
                .show(ui, |ui| {
                    ui.set_max_width((ui.available_width() * 0.78).max(200.0));
                    ui.label(RichText::new(text).size(14.5).color(fg));
                });
        });
    }

    fn tools_row(&mut self, ui: &mut egui::Ui, theme: &Theme) {
        let p = theme.pal;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 10.0;
            ui.spacing_mut().interact_size.y = 28.0;
            theme::switch(ui, theme, &mut self.agent, tr("Агент"));
            theme::switch(ui, theme, &mut self.think, "Think");
            theme::switch(ui, theme, &mut self.web, "Web");
            if self.web {
                // Движок поиска — тот же набор, что в настройках браузера.
                egui::ComboBox::from_id_salt("web_engine")
                    .selected_text(
                        RichText::new(engine_label(&self.web_engine))
                            .font(theme::title_font(13.0)),
                    )
                    .width(118.0)
                    .show_ui(ui, |ui| {
                        for (id, name) in [
                            ("duckduckgo", "DuckDuckGo"),
                            ("bing", "Bing"),
                            ("google", "Google"),
                            ("yandex", tr("Яндекс")),
                        ] {
                            ui.selectable_value(&mut self.web_engine, id.to_string(), name);
                        }
                    });
            }
            if !self.web_note.is_empty() {
                ui.add_space(4.0);
                ui.label(
                    RichText::new(&self.web_note)
                        .size(12.0)
                        .color(p.on_surface_variant),
                );
            }
        });
    }

    fn composer(
        &mut self,
        ui: &mut egui::Ui,
        theme: &Theme,
        python: &str,
        engine_dir: &Path,
        ws: &Path,
    ) {
        let p = theme.pal;
        let busy = self.is_busy();
        // M3 search bar: тональная капсула 28dp без обводки.
        egui::Frame::none()
            .fill(p.surface_container)
            .rounding(Rounding::same(4.0))
            .inner_margin(egui::Margin::symmetric(16.0, 10.0))
            .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.add(
                egui::TextEdit::multiline(&mut self.input)
                    .desired_rows(1)
                    .desired_width(f32::INFINITY)
                    .frame(false)
                    .text_color(p.on_surface)
                    .hint_text(if self.agent {
                        tr("Что сделать в проекте…")
                    } else {
                        tr("Спросить что угодно")
                    })
                    .margin(egui::Margin::symmetric(2.0, 4.0)),
            );
            ui.add_space(2.0);
            ui.horizontal(|ui| {
                if self.agent_running {
                    if theme::button(ui, theme, tr("Стоп"), ButtonKind::Tonal, true).clicked() {
                        self.stop_agent();
                    }
                    ui.add(egui::Spinner::new().size(15.0).color(p.primary));
                    ui.label(theme.label(&self.agent_meta));
                } else {
                match self.status {
                    Status::Failed => {
                        if theme::button(ui, theme, tr("Повторить загрузку"), ButtonKind::Tonal, !busy).clicked() {
                            self.load(python, engine_dir, ws);
                        }
                    }
                    Status::Generating => {
                        if theme::button(ui, theme, tr("Стоп"), ButtonKind::Tonal, true).clicked() {
                            self.stop();
                        }
                        ui.add(egui::Spinner::new().size(15.0).color(p.primary));
                    }
                    Status::Ready => {
                        let name = self.model.rsplit('/').next().unwrap_or(&self.model).to_string();
                        theme::chip(ui, theme, &name);
                    }
                    Status::Loading => {
                        ui.add(egui::Spinner::new().size(15.0).color(p.primary));
                        ui.label(theme.label(tr("загружаю модель…")));
                    }
                    Status::Unloaded => {
                        theme::chip(ui, theme, tr("модель не выбрана"));
                    }
                }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let can = self.status == Status::Ready
                        && !self.input.trim().is_empty()
                        && !self.agent_running;
                    let (fill, fg) = if can {
                        (p.primary, p.on_primary)
                    } else {
                        (p.surface_container_highest, p.on_surface_variant)
                    };
                    let (rect, resp) =
                        ui.allocate_exact_size(egui::vec2(40.0, 40.0), egui::Sense::click());
                    let btn_fill = if resp.hovered() && can {
                        theme::lerp_color(fill, fg, 0.12)
                    } else {
                        fill
                    };
                    ui.painter().rect_filled(
                        rect.shrink(2.0),
                        Rounding::same(10.0),
                        btn_fill,
                    );
                    theme::draw_icon(
                        ui.painter(),
                        rect.center(),
                        theme::icons::ARROW_UP,
                        20.0,
                        fg,
                    );
                    if resp.clicked() && can {
                        self.ask();
                    }
                });
            });
        });
    }

    /// Параметры запуска — M3-диалог поверх чата, чтобы развёрнутая
    /// форма не выдавливала композер за пределы окна.
    fn advanced_window(
        &mut self,
        ctx: &egui::Context,
        theme: &Theme,
        python: &str,
        engine_dir: &Path,
        ws: &Path,
        busy: bool,
    ) {
        if !self.show_adv {
            return;
        }
        let p = theme.pal;
        let mut open = true;
        let mut reload = false;
        let mut close_clicked = false;
        let mut api_dirty = false;
        egui::Window::new(tr("Параметры запуска"))
            .open(&mut open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
            .default_width(520.0)
            .frame(
                egui::Frame::none()
                    .fill(p.surface_container_high)
                    .rounding(Rounding::same(28.0))
                    .inner_margin(egui::Margin::same(20.0))
                    .shadow(egui::epaint::Shadow {
                        offset: egui::vec2(0.0, 8.0),
                        blur: 24.0,
                        spread: 0.0,
                        color: egui::Color32::from_black_alpha(80),
                    }),
            )
            .show(ctx, |ui| {
                ui.set_width(500.0);
                ui.horizontal_wrapped(|ui| {
                    ui.label(theme.label(tr("Контекст")));
                    small_field(ui, theme, &mut self.ctx, "2048");
                    ui.label(theme.label(tr("KV-кэш")));
                    combo_str(ui, theme, "kv", &mut self.kv_type, &["f16", "q8_0", "q4_0"]);
                    ui.label(theme.label(tr("Потоки")));
                    small_field(ui, theme, &mut self.threads, "8");
                    ui.label(theme.label(tr("Макс. токенов")));
                    small_field(ui, theme, &mut self.max_tokens, "256");
                    ui.label(theme.label("temp"));
                    small_field(ui, theme, &mut self.temp, "0.7");
                });
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label(theme.label(tr("LoRA-адаптер")));
                    let current = if self.adapter.is_empty() {
                        tr("без адаптера").to_string()
                    } else {
                        self.adapter.clone()
                    };
                    egui::ComboBox::from_id_salt("adapter")
                        .selected_text(current)
                        .show_ui(ui, |ui| {
                            for o in &self.adapters {
                                let label = if o.is_empty() {
                                    tr("без адаптера").to_string()
                                } else {
                                    o.clone()
                                };
                                ui.selectable_value(&mut self.adapter, o.clone(), label);
                            }
                        });
                });
                if self.agent {
                    ui.add_space(6.0);
                    ui.label(theme.section(tr("Папка проекта")));
                    ui.add(
                        egui::TextEdit::singleline(&mut self.agent_root)
                            .desired_width(f32::INFINITY)
                            .hint_text(self.workspace.clone()),
                    );
                }
                ui.add_space(12.0);
                ui.label(theme.section(tr("Публичный API")));
                if ui.checkbox(&mut self.api_enabled, tr("Открыть доступ к текущей модели")).changed() {
                    api_dirty = true;
                }
                if self.api_enabled {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(theme.label(tr("Порт")));
                        let before = self.api_port.clone();
                        small_field(ui, theme, &mut self.api_port, "8000");
                        if self.api_port != before {
                            api_dirty = true;
                        }
                        ui.label(theme.label(tr("Ключ")));
                        ui.label(theme.mono(self.api_key.clone(), 12.0));
                        ui.label(theme.label("IP"));
                        let host_before = self.api_host.clone();
                        wide_field(ui, theme, &mut self.api_host, tr("авто: Radmin"), 120.0);
                        if self.api_host != host_before {
                            api_dirty = true;
                        }
                    });
                    if self.api_running {
                        ui.horizontal_wrapped(|ui| {
                            ui.label(theme.label(self.api_url.clone()));
                            if theme::button(ui, theme, tr("Копировать"), ButtonKind::Outlined, true).clicked() {
                                ui.output_mut(|o| o.copied_text = self.api_url.clone());
                            }
                        });
                        ui.label(theme.label(tr("Модель: current-activated · system prompt: имя модели")));
                    } else {
                        ui.label(theme.label(tr("Загрузи модель — API поднимется автоматически.")));
                    }
                    if !self.api_note.is_empty() {
                        ui.label(theme.label(self.api_note.clone()));
                    }
                    ui.label(theme.label(tr("Доступ по Radmin VPN. Ключ 1, без HTTPS — не выставляй порт в интернет.")));
                }
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if theme::button(ui, theme, tr("Загрузить модель"), ButtonKind::Filled, !busy).clicked() {
                        reload = true;
                    }
                    if theme::button(ui, theme, tr("Закрыть"), ButtonKind::Text, true).clicked() {
                        close_clicked = true;
                    }
                });
            });
        if close_clicked {
            open = false;
        }
        self.show_adv = open;
        if api_dirty {
            self.api_apply();
        }
        if reload {
            self.load(python, engine_dir, ws);
        }
    }
}

impl Drop for Chat {
    fn drop(&mut self) {
        if let Some(r) = self.engine.as_mut() {
            let _ = r.send("{\"cmd\":\"shutdown\"}");
            let _ = r.wait_exit(std::time::Duration::from_secs(3));
            r.kill();
        }
    }
}

fn small_field(ui: &mut egui::Ui, theme: &Theme, value: &mut String, hint: &str) {
    wide_field(ui, theme, value, hint, 56.0);
}

fn wide_field(ui: &mut egui::Ui, theme: &Theme, value: &mut String, hint: &str, width: f32) {
    let p = theme.pal;
    egui::Frame::none()
        .fill(p.surface_container_highest)
        .rounding(Rounding::same(12.0))
        .inner_margin(egui::Margin::symmetric(10.0, 5.0))
        .show(ui, |ui| {
            ui.add(
                egui::TextEdit::singleline(value)
                    .frame(false)
                    .desired_width(width)
                    .text_color(p.on_surface)
                    .hint_text(hint),
            );
        });
}

fn combo_str(ui: &mut egui::Ui, theme: &Theme, id: &str, value: &mut String, options: &[&str]) {
    let p = theme.pal;
    egui::Frame::none()
        .fill(p.surface_container_highest)
        .rounding(Rounding::same(12.0))
        .inner_margin(egui::Margin::symmetric(10.0, 4.0))
        .show(ui, |ui| {
            egui::ComboBox::from_id_salt(id)
                .selected_text(value.clone())
                .width(80.0)
                .show_ui(ui, |ui| {
                    for o in options {
                        ui.selectable_value(value, (*o).to_string(), *o);
                    }
                });
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn groups_split_gguf_shards() {
        let (base, part, total) = split_shard("Forge-9B-Q4-00001-of-00003.gguf").unwrap();
        assert_eq!(base, "forge-9b-q4");
        assert_eq!((part, total), (1, 3));
        assert!(split_shard("plain-model.gguf").is_none());
        assert!(split_shard("fake-00000-of-00002.gguf").is_none());
    }

    #[test]
    fn scan_lists_split_model_once() {
        let dir = std::env::temp_dir().join(format!("lf_split_test_{}", std::process::id()));
        let gg = dir.join("models_gguf");
        std::fs::create_dir_all(&gg).unwrap();
        std::fs::write(gg.join("m-Q4-00001-of-00002.gguf"), b"aa").unwrap();
        std::fs::write(gg.join("m-Q4-00002-of-00002.gguf"), b"bbbb").unwrap();
        let (models, _) = scan_workspace(&dir);
        let found: Vec<_> = models.iter().filter(|m| m.path.ends_with(".gguf")).collect();
        assert_eq!(found.len(), 1);
        assert!(found[0].path.ends_with("00001-of-00002.gguf"));
        std::fs::remove_dir_all(&dir).ok();
    }

    fn wait_for(chat: &mut Chat, want: Status, secs: u64) -> bool {
        let deadline = Instant::now() + Duration::from_secs(secs);
        while Instant::now() < deadline {
            chat.poll();
            if chat.status == want {
                return true;
            }
            if chat.status == Status::Failed {
                return false;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        false
    }

    #[test]
    #[ignore]
    fn real_chat_round_trip() {
        let ws = std::path::Path::new("..");
        let engine = std::path::Path::new("python");
        let mut chat = Chat::new(ws);
        chat.model = "models_gguf/Forge-1-E2B-Q4_K_M.gguf".into();
        chat.ctx = "1024".into();
        chat.max_tokens = "8".into();
        chat.load("python", engine, ws);
        if !wait_for(&mut chat, Status::Ready, 240) {
            eprintln!("{}", crate::tf!("STDERR движка: {:?}", "Engine STDERR: {:?}", chat.log.iter().collect::<Vec<_>>()));
            panic!("{} {:?}", tr("модель не загрузилась:"), chat.error);
        }
        chat.input = "Say hi".into();
        chat.ask();
        assert!(wait_for(&mut chat, Status::Ready, 120), "{} {:?}", tr("нет ответа:"), chat.error);
        let answer = chat.messages.last().map(|m| m.content.clone()).unwrap_or_default();
        assert!(!answer.trim().is_empty(), "{}", tr("пустой ответ"));
        eprintln!("{} {}", tr("ответ:"), answer);
    }

    #[test]
    fn active_hint_only_for_e_models() {
        assert_eq!(active_hint("Forge-1-E2B-Q4_K_M").as_deref(), Some(tr("2B акт.")));
        assert_eq!(active_hint("Qwen2.5-1.5B-Instruct"), None);
        assert_eq!(active_hint("train.jsonl"), None);
    }
}
