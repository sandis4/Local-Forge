//! GUI на egui в стиле Material 3 (Material You, как на Pixel).

use crate::i18n::tr;
use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, TryRecvError};

use eframe::egui::{self, RichText, ScrollArea};

use crate::chat::Chat;
use crate::downloads::Downloads;
use crate::folders::{FolderPicker, PickerResult};
use crate::parse::{human_duration, Stats};
use crate::runner::Runner;
use crate::theme::{self, ButtonKind, Theme};
use crate::validate::{self, Report, Settings};
use crate::Launch;

const LOG_LIMIT: usize = 600;

#[derive(PartialEq, Clone, Copy)]
enum Tab {
    Chat,
    Train,
    Download,
    Settings,
}

fn tab_label(tab: Tab) -> &'static str {
    match tab {
        Tab::Chat => tr("Чат"),
        Tab::Train => tr("Обучение"),
        Tab::Download => tr("Хаб моделей"),
        Tab::Settings => tr("Настройки"),
    }
}


#[derive(PartialEq, Clone, Copy)]
enum Phase {
    Setup,
    Running,
    Done,
}

enum CheckState {
    Idle,
    Running(Receiver<Result<Report, String>>),
    Done(Report),
    Failed(String),
}

#[derive(Clone)]
struct Form {
    model: String,
    train_file: String,
    val_file: String,
    out: String,
    identity_file: String,
    epochs: String,
    lr: String,
    max_grad_norm: String,
    warmup_ratio: String,
    max_length: String,
    max_samples: String,
    batch_size: String,
    grad_accum: String,
    save_steps: String,
    lora_r: String,
    lora_alpha: String,
    lora_plus_ratio: String,
    tune_mode: String,
    preset: String,
    dtype: String,
    amp: bool,
    amp_dtype: String,
    grad_checkpointing: bool,
    optimizer: String,
    lisa_layers: String,
    lisa_interval: String,
    quant4: String,
    attn: String,
    liger: bool,
    compile: bool,
    replay_file: String,
    replay_ratio: String,
    kl_anchor: String,
    objective: String,
    orpo_file: String,
    orpo_beta: String,
    heretic_trials: String,
    heretic_startup_trials: String,
    heretic_quant: String,
    group_by_length: bool,
    packing: bool,
    kd: bool,
    kd_weight: String,
    kd_top_k: String,
    cache: bool,
    extra: String,
}

fn num_or<T: std::str::FromStr>(s: &str, default: T) -> T {
    s.trim().parse().unwrap_or(default)
}

fn opt_num<T: std::str::FromStr>(s: &str) -> Option<T> {
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        t.parse().ok()
    }
}

impl Form {
    fn from_launch(l: &Launch) -> Self {
        Self {
            model: l.model.clone(),
            train_file: l.data.clone(),
            val_file: l.val_file.clone(),
            out: l.out.clone(),
            identity_file: l.identity_file.clone(),
            epochs: l.epochs.to_string(),
            lr: format!("{}", l.lr),
            max_grad_norm: format!("{}", l.max_grad_norm),
            warmup_ratio: format!("{}", l.warmup_ratio),
            max_length: l.max_length.map(|v| v.to_string()).unwrap_or_default(),
            max_samples: l.max_samples.map(|v| v.to_string()).unwrap_or_default(),
            batch_size: l.batch_size.map(|v| v.to_string()).unwrap_or_default(),
            grad_accum: l.grad_accum.map(|v| v.to_string()).unwrap_or_default(),
            save_steps: l.save_steps.map(|v| v.to_string()).unwrap_or_default(),
            lora_r: l.lora_r.map(|v| v.to_string()).unwrap_or_default(),
            lora_alpha: l.lora_alpha.map(|v| v.to_string()).unwrap_or_default(),
            lora_plus_ratio: format!("{}", l.lora_plus_ratio),
            tune_mode: l.tune_mode.clone(),
            preset: l.preset.clone(),
            dtype: l.dtype.clone(),
            amp: l.amp,
            amp_dtype: l.amp_dtype.clone().unwrap_or_else(|| "auto".into()),
            grad_checkpointing: l.grad_checkpointing,
            optimizer: l.optimizer.clone().unwrap_or_else(|| "auto".into()),
            lisa_layers: l.lisa_layers.map(|v| v.to_string()).unwrap_or_default(),
            lisa_interval: l.lisa_interval.map(|v| v.to_string()).unwrap_or_default(),
            quant4: l.quant4.clone(),
            attn: l.attn.clone(),
            liger: l.liger,
            compile: l.compile,
            replay_file: l.replay_file.clone(),
            replay_ratio: format!("{}", l.replay_ratio),
            kl_anchor: format!("{}", l.kl_anchor),
            objective: l.objective.clone(),
            orpo_file: l.orpo_file.clone(),
            orpo_beta: format!("{}", l.orpo_beta),
            heretic_trials: l.heretic_trials.to_string(),
            heretic_startup_trials: l.heretic_startup_trials.to_string(),
            heretic_quant: l.heretic_quant.clone(),
            group_by_length: l.group_by_length,
            packing: l.packing,
            kd: l.kd,
            kd_weight: format!("{}", l.kd_weight),
            kd_top_k: l.kd_top_k.to_string(),
            cache: l.cache,
            extra: l.extra.join(" "),
        }
    }

    fn settings(&self) -> Settings {
        Settings {
            model: self.model.trim().to_string(),
            train_file: self.train_file.trim().to_string(),
            val_file: self.val_file.trim().to_string(),
            out: self.out.trim().to_string(),
            identity_file: self.identity_file.trim().to_string(),
            epochs: num_or(&self.epochs, 1),
            lr: num_or(&self.lr, 2e-5),
            max_grad_norm: num_or(&self.max_grad_norm, 1.0),
            warmup_ratio: num_or(&self.warmup_ratio, 0.05),
            max_length: num_or(&self.max_length, 512),
            max_samples: opt_num(&self.max_samples),
            batch_size: num_or(&self.batch_size, 1),
            grad_accum: num_or(&self.grad_accum, 8),
            save_steps: num_or(&self.save_steps, 50),
            lora_r: num_or(&self.lora_r, 8),
            lora_alpha: num_or(&self.lora_alpha, 16),
            lora_plus_ratio: num_or(&self.lora_plus_ratio, 0.0),
            tune_mode: self.tune_mode.clone(),
            preset: self.preset.clone(),
            dtype: self.dtype.clone(),
            amp: self.amp,
            amp_dtype: self.amp_dtype.clone(),
            grad_checkpointing: self.grad_checkpointing,
            optimizer: self.optimizer.clone(),
            lisa_layers: num_or(&self.lisa_layers, 0),
            lisa_interval: num_or(&self.lisa_interval, 50),
            quant4: self.quant4.clone(),
            attn: self.attn.clone(),
            liger: self.liger,
            compile: self.compile,
            replay_file: self.replay_file.trim().to_string(),
            replay_ratio: num_or(&self.replay_ratio, 0.2),
            kl_anchor: num_or(&self.kl_anchor, 0.0),
            objective: self.objective.clone(),
            orpo_file: self.orpo_file.trim().to_string(),
            orpo_beta: num_or(&self.orpo_beta, 0.1),
            group_by_length: self.group_by_length,
            packing: self.packing,
            kd: self.kd,
            kd_weight: num_or(&self.kd_weight, 0.5),
            kd_top_k: num_or(&self.kd_top_k, 8),
            cache: self.cache,
        }
    }

    fn launch(&self, base: &Launch) -> Launch {
        // Общие поля берём из settings() — единый источник значений формы.
        // Поля «авто» (пусто/None) остаются None: тренер сам выберет значение.
        let s = self.settings();
        let mut l = base.clone();
        l.model = s.model;
        l.data = s.train_file;
        l.val_file = s.val_file;
        l.identity_file = s.identity_file;
        l.out = s.out;
        l.preset = s.preset;
        l.epochs = s.epochs;
        l.lr = s.lr;
        l.max_grad_norm = s.max_grad_norm;
        l.warmup_ratio = s.warmup_ratio;
        l.save_steps = opt_num(&self.save_steps);
        l.max_samples = s.max_samples;
        l.max_length = opt_num(&self.max_length);
        l.batch_size = opt_num(&self.batch_size);
        l.grad_accum = opt_num(&self.grad_accum);
        l.lora_r = opt_num(&self.lora_r);
        l.lora_alpha = opt_num(&self.lora_alpha);
        l.lora_plus_ratio = s.lora_plus_ratio;
        l.tune_mode = s.tune_mode;
        l.dtype = s.dtype;
        l.amp = s.amp;
        l.amp_dtype = if s.amp_dtype == "auto" {
            None
        } else {
            Some(s.amp_dtype)
        };
        l.grad_checkpointing = s.grad_checkpointing;
        l.optimizer = if s.optimizer == "auto" {
            None
        } else {
            Some(s.optimizer)
        };
        l.lisa_layers = opt_num(&self.lisa_layers);
        l.lisa_interval = opt_num(&self.lisa_interval);
        l.quant4 = s.quant4;
        l.attn = s.attn;
        l.liger = s.liger;
        l.compile = s.compile;
        l.replay_file = s.replay_file;
        l.replay_ratio = s.replay_ratio;
        l.kl_anchor = s.kl_anchor;
        l.objective = s.objective;
        l.orpo_file = s.orpo_file;
        l.orpo_beta = s.orpo_beta;
        l.heretic_trials = num_or(&self.heretic_trials, 3);
        l.heretic_startup_trials = num_or(&self.heretic_startup_trials, 1);
        l.heretic_quant = self.heretic_quant.clone();
        l.group_by_length = s.group_by_length;
        l.packing = s.packing;
        l.kd = s.kd;
        l.kd_weight = s.kd_weight;
        l.kd_top_k = s.kd_top_k;
        l.cache = s.cache;
        l.extra = self.extra.split_whitespace().map(|s| s.to_string()).collect();
        l
    }
}

pub struct GuiApp {
    theme: Theme,
    base: Launch,
    form: Form,
    check: CheckState,
    checked_sig: Option<Settings>,
    plan_applied: bool,
    pending_start: bool,
    scroll_top: bool,
    initializing: bool,
    splash_until: std::time::Instant,
    splash_alpha: f32,
    splash_fading: bool,
    theme_dark: bool,
    seed: [u8; 3],
    settings: crate::settings::AppSettings,
    theme_anim: Option<(crate::theme::Palette, f32)>,
    content_alpha: f32,
    folder_pick: Option<FolderPicker>,
    cmd_input: String,
    cmd_status: String,
    tab: Tab,
    chat: Chat,
    downloads: Downloads,
    phase: Phase,
    runner: Option<Runner>,
    stats: Stats,
    log: VecDeque<String>,
    paused: bool,
    quitting: bool,
    exit_code: Option<i32>,
    error: Option<String>,
}

const BOOL_FLAGS: &[&str] = &[
    "amp", "no_amp", "no_grad_checkpointing", "gradient_checkpointing",
    "liger", "compile", "packing", "group_by_length", "cache", "no_cache",
    "auto", "no_auto", "no_state_save", "no_resume_state", "kd",
];

/// Разбивает строку команды на флаги с учётом кавычек и `--flag=value`.
fn command_flags(text: &str) -> Vec<(String, Option<String>)> {
    let mut tokens: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for ch in text.chars() {
        match ch {
            '"' | '\'' if quote.is_none() => quote = Some(ch),
            c if Some(c) == quote => quote = None,
            c if c.is_whitespace() && quote.is_none() => {
                if !cur.is_empty() {
                    tokens.push(std::mem::take(&mut cur));
                }
            }
            c => cur.push(c),
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }

    let mut out = Vec::new();
    let mut i = 0;
    while i < tokens.len() && !tokens[i].starts_with("--") {
        i += 1;
    }
    while i < tokens.len() {
        let token = &tokens[i];
        if !token.starts_with("--") {
            i += 1;
            continue;
        }
        let body = token.trim_start_matches("--");
        let (key, inline) = match body.split_once('=') {
            Some((k, v)) => (k.to_string(), Some(v.to_string())),
            None => (body.to_string(), None),
        };
        let key = key.to_lowercase().replace('-', "_");
        let is_bool = BOOL_FLAGS.contains(&key.as_str());
        let mut value = inline;
        if value.is_none()
            && !is_bool
            && i + 1 < tokens.len()
            && !tokens[i + 1].starts_with("--")
        {
            value = Some(tokens[i + 1].clone());
            i += 1;
        }
        out.push((key, value));
        i += 1;
    }
    out
}

/// Применяет флаги main.py к форме. Возвращает (применено, в extra, ошибки).
fn apply_flags(form: &mut Form, text: &str) -> (usize, usize, Vec<String>) {
    let flags = command_flags(text);
    let mut applied = 0usize;
    let mut problems = Vec::new();
    let mut extra: Vec<String> = Vec::new();

    for (key, value) in flags {
        let v = value.clone().unwrap_or_default();
        let need = |v: &str, k: &str, problems: &mut Vec<String>| -> Option<String> {
            if v.is_empty() {
                problems.push(crate::tf!("--{k} без значения", "--{k} without a value"));
                None
            } else {
                Some(v.to_string())
            }
        };
        if let Some(slot) = form_text_field(form, &key) {
            if let Some(x) = need(&v, &key, &mut problems) {
                *slot = x;
                applied += 1;
            }
            continue;
        }
        let flag: Option<(&mut bool, bool)> = match key.as_str() {
            "amp" => Some((&mut form.amp, true)),
            "no_amp" => Some((&mut form.amp, false)),
            "no_grad_checkpointing" => Some((&mut form.grad_checkpointing, false)),
            "gradient_checkpointing" => Some((&mut form.grad_checkpointing, true)),
            "liger" => Some((&mut form.liger, true)),
            "compile" => Some((&mut form.compile, true)),
            "packing" => Some((&mut form.packing, true)),
            "group_by_length" => Some((&mut form.group_by_length, true)),
            "kd" => Some((&mut form.kd, true)),
            "cache" => Some((&mut form.cache, true)),
            "no_cache" => Some((&mut form.cache, false)),
            _ => None,
        };
        if let Some((slot, on)) = flag {
            *slot = on;
            applied += 1;
            continue;
        }
        let mut piece = format!("--{key}");
        if let Some(v) = value {
            if v.contains(' ') {
                piece.push_str(&format!(" \"{v}\""));
            } else {
                piece.push(' ');
                piece.push_str(&v);
            }
        }
        extra.push(piece);
        applied += 1;
    }

    form.extra = extra.join(" ");
    (applied - extra.len(), extra.len(), problems)
}

/// Строковые поля формы и их алиасы из аргументов main.py.
fn form_text_field<'a>(form: &'a mut Form, key: &str) -> Option<&'a mut String> {
    match key {
        "model" => Some(&mut form.model),
        "train_file" | "data" => Some(&mut form.train_file),
        "val_file" | "val" => Some(&mut form.val_file),
        "output_dir" | "out" => Some(&mut form.out),
        "identity_file" => Some(&mut form.identity_file),
        "epochs" => Some(&mut form.epochs),
        "lr" => Some(&mut form.lr),
        "max_grad_norm" => Some(&mut form.max_grad_norm),
        "warmup_ratio" => Some(&mut form.warmup_ratio),
        "max_length" => Some(&mut form.max_length),
        "max_samples" => Some(&mut form.max_samples),
        "batch_size" => Some(&mut form.batch_size),
        "grad_accum" => Some(&mut form.grad_accum),
        "save_steps" => Some(&mut form.save_steps),
        "lora_r" => Some(&mut form.lora_r),
        "lora_alpha" => Some(&mut form.lora_alpha),
        "lora_plus_ratio" => Some(&mut form.lora_plus_ratio),
        "tune_mode" => Some(&mut form.tune_mode),
        "preset" => Some(&mut form.preset),
        "dtype" => Some(&mut form.dtype),
        "amp_dtype" => Some(&mut form.amp_dtype),
        "optimizer" => Some(&mut form.optimizer),
        "lisa_layers" => Some(&mut form.lisa_layers),
        "lisa_interval" => Some(&mut form.lisa_interval),
        "quant4" => Some(&mut form.quant4),
        "attn" => Some(&mut form.attn),
        "replay_file" => Some(&mut form.replay_file),
        "replay_ratio" => Some(&mut form.replay_ratio),
        "kl_anchor" => Some(&mut form.kl_anchor),
        "objective" => Some(&mut form.objective),
        "orpo_file" => Some(&mut form.orpo_file),
        "orpo_beta" => Some(&mut form.orpo_beta),
        "heretic_trials" => Some(&mut form.heretic_trials),
        "heretic_startup_trials" => Some(&mut form.heretic_startup_trials),
        "heretic_quant" => Some(&mut form.heretic_quant),
        "kd_weight" => Some(&mut form.kd_weight),
        "kd_top_k" => Some(&mut form.kd_top_k),
        _ => None,
    }
}

fn warn_color() -> egui::Color32 {
    egui::Color32::from_rgb(0xFF, 0xB4, 0x4D)
}

fn ok_color() -> egui::Color32 {
    egui::Color32::from_rgb(0x6D, 0xD5, 0x8C)
}

impl GuiApp {
    pub fn new(cc: &eframe::CreationContext<'_>, base: Launch) -> Self {
        Theme::install_fonts(&cc.egui_ctx);
        let theme = Theme::new(false);
        theme.apply(&cc.egui_ctx);

        let form = Form::from_launch(&base);
        let mut chat = Chat::new(&base.ws);
        chat.agent_root = base.ws.display().to_string();
        let app = Self {
            theme,
            base,
            form,
            check: CheckState::Idle,
            checked_sig: None,
            plan_applied: false,
            pending_start: false,
            scroll_top: true,
            initializing: true,
            splash_until: std::time::Instant::now() + std::time::Duration::from_millis(420),
            splash_alpha: 1.0,
            splash_fading: false,
            theme_dark: false,
            seed: crate::color::DEFAULT_SEED,
            settings: crate::settings::AppSettings::default(),
            theme_anim: None,
            content_alpha: 1.0,
            folder_pick: None,
            cmd_input: String::new(),
            cmd_status: String::new(),
            tab: Tab::Chat,
            chat,
            downloads: Downloads::new(),
            phase: Phase::Setup,
            runner: None,
            stats: Stats::default(),
            log: VecDeque::new(),
            paused: false,
            quitting: false,
            exit_code: None,
            error: None,
        };
        // Никакого Python при старте: окно появляется сразу.
        // Проверка запускается при входе на «Обучение» или по кнопке.
        app
    }

    fn add_project(&mut self, path: String) {
        if !self.settings.projects.contains(&path) {
            self.settings.projects.push(path.clone());
        }
        self.set_project(path);
    }

    fn set_project(&mut self, path: String) {
        self.settings.active_project = path.clone();
        self.settings.agent_root = path.clone();
        self.chat.agent_root = path;
        self.persist_settings();
    }

    fn remove_project(&mut self, path: &str) {
        self.settings.projects.retain(|p| p != path);
        if self.settings.active_project == path {
            self.settings.active_project.clear();
            self.settings.agent_root.clear();
            self.chat.agent_root.clear();
        }
        self.persist_settings();
    }

    fn tab_title(&self) -> &'static str {
        tab_label(self.tab)
    }

    /// M3 top app bar: заголовок экрана, перетаскивание и кнопки окна.
    fn draw_titlebar(&mut self, ctx: &egui::Context) {
        let p = self.theme.pal;
        let maximized = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
        egui::TopBottomPanel::top("titlebar")
            .exact_height(48.0)
            .frame(
                egui::Frame::none()
                    .fill(p.surface_container_low)
                    .inner_margin(egui::Margin::symmetric(16.0, 0.0)),
            )
            .show(ctx, |ui| {
                let full = ui.max_rect();
                ui.painter().line_segment(
                    [
                        egui::pos2(full.left(), full.bottom() - 0.5),
                        egui::pos2(full.right(), full.bottom() - 0.5),
                    ],
                    egui::Stroke::new(1.0_f32, p.outline_variant),
                );
                ui.horizontal_centered(|ui| {
                    let (mark, _) =
                        ui.allocate_exact_size(egui::vec2(8.0, 8.0), egui::Sense::hover());
                    ui.painter().rect_filled(mark, egui::Rounding::ZERO, p.primary);
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new("LOCAL FORGE")
                            .font(egui::FontId::new(10.5, egui::FontFamily::Monospace))
                            .color(p.on_surface_variant),
                    );
                    ui.add_space(10.0);
                    ui.label(
                        RichText::new("/")
                            .font(egui::FontId::new(11.0, egui::FontFamily::Monospace))
                            .color(p.outline),
                    );
                    ui.add_space(10.0);
                    ui.label(
                        RichText::new(self.tab_title())
                            .font(theme::title_font(17.0))
                            .color(p.on_surface),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if window_button(ui, &self.theme, WindowButton::Close).clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                        if window_button(ui, &self.theme, WindowButton::Maximize(maximized)).clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized));
                        }
                        if window_button(ui, &self.theme, WindowButton::Minimize).clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true));
                        }
                        ui.add_space(8.0);

                        // Пустое место шапки — область перетаскивания окна.
                        let drag = egui::Sense::click_and_drag();
                        let (rect, resp) = ui.allocate_exact_size(ui.available_size(), drag);
                        if resp.double_clicked() {
                            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized));
                        } else if resp.drag_started()
                            && ui.input(|i| i.pointer.hover_pos().map_or(true, |q| q.y > rect.top() + 6.0))
                        {
                            ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                        }
                    });
                });
            });
    }

    /// Сплэш поверх интерфейса: при завершении плавно растворяется,
    /// а не исчезает рывком.
    fn draw_splash(&self, ctx: &egui::Context) {
        let p = self.theme.pal;
        let screen = ctx.screen_rect();
        let a = self.splash_alpha.clamp(0.0, 1.0);
        let bg = egui::Color32::from_rgba_unmultiplied(
            p.surface.r(),
            p.surface.g(),
            p.surface.b(),
            (255.0 * a) as u8,
        );
        egui::Area::new(egui::Id::new("splash"))
            .order(egui::Order::Foreground)
            .fixed_pos(screen.min)
            .show(ctx, |ui| {
                ui.set_min_size(screen.size());
                ui.painter().rect_filled(screen, 0.0, bg);
                ui.multiply_opacity(a);
                ui.add_space((screen.height() * 0.33).min(220.0));
                ui.vertical_centered(|ui| {
                    let (rect, _) =
                        ui.allocate_exact_size(egui::vec2(76.0, 76.0), egui::Sense::hover());
                    let painter = ui.painter();
                    painter.circle_filled(rect.center(), 38.0, p.primary_container);
                    painter.text(
                        rect.center(),
                        egui::Align2::CENTER_CENTER,
                        "LF",
                        egui::FontId::new(27.0, theme::bold_family()),
                        p.on_primary_container,
                    );
                    ui.add_space(16.0);
                    ui.label(
                        RichText::new("Local Forge")
                            .font(egui::FontId::new(28.0, theme::bold_family()))
                            .color(p.on_surface),
                    );
                    ui.add_space(14.0);
                    ui.add(egui::Spinner::new().size(22.0).color(p.primary));
                    ui.add_space(4.0);
                    ui.label(self.theme.label(tr("Запуск…")));
                });
            });
    }

    /// Переключатели режимов чата сохраняются сразу, без экрана настроек.
    fn sync_chat_settings(&mut self) {
        let c = &self.chat;
        let changed = self.settings.chat_agent != c.agent
            || self.settings.chat_think != c.think
            || self.settings.chat_web != c.web
            || self.settings.chat_web_engine != c.web_engine
            || self.settings.chat_api != c.api_enabled
            || self.settings.chat_api_port != c.api_port.trim().parse().unwrap_or(8000)
            || self.settings.chat_api_host != c.api_host
            || self.settings.agent_root != c.agent_root;
        if !changed {
            return;
        }
        self.settings.chat_agent = c.agent;
        self.settings.chat_think = c.think;
        self.settings.chat_web = c.web;
        self.settings.chat_web_engine = c.web_engine.clone();
        self.settings.chat_api = c.api_enabled;
        self.settings.chat_api_port = c.api_port.trim().parse().unwrap_or(8000);
        self.settings.chat_api_host = c.api_host.clone();
        self.settings.agent_root = c.agent_root.clone();
        self.persist_settings();
    }

    fn persist_settings(&mut self) {
        self.settings.dark = self.theme_dark;
        crate::settings::save(&self.settings);
    }

    /// Применить настройки к чату и путям.
    fn apply_settings(&mut self) {
        self.chat.threads = self.settings.chat_threads.to_string();
        self.chat.ctx = self.settings.chat_ctx.to_string();
        self.chat.kv_type = self.settings.chat_kv.clone();
        self.chat.moe_lazy = self.settings.chat_moe_lazy;
        self.chat.max_tokens = self.settings.chat_max_tokens.to_string();
        self.chat.temp = format!("{}", self.settings.chat_temp);
        self.chat.agent = self.settings.chat_agent;
        self.chat.think = self.settings.chat_think;
        self.chat.web = self.settings.chat_web;
        self.chat.api_enabled = self.settings.chat_api;
        self.chat.api_port = self.settings.chat_api_port.to_string();
        self.chat.api_host = self.settings.chat_api_host.clone();
        if !self.settings.chat_web_engine.trim().is_empty() {
            self.chat.web_engine = self.settings.chat_web_engine.clone();
        }
        if !self.settings.agent_root.trim().is_empty() {
            self.chat.agent_root = self.settings.agent_root.clone();
        }
        if !self.settings.active_project.trim().is_empty() {
            self.chat.agent_root = self.settings.active_project.clone();
        }
        if !self.settings.python.trim().is_empty() {
            self.base.python = self.settings.python.trim().to_string();
        }
        if !self.settings.engine.trim().is_empty() {
            self.base.engine = std::path::PathBuf::from(self.settings.engine.trim());
        }
    }

    fn set_theme(&mut self, dark: bool) {
        self.theme_dark = dark;
        self.theme = Theme::from_seed(self.seed, dark);
        self.theme_anim = None;
    }

    /// Плавная смена темы: интерполируем палитру, а не переключаем рывком.
    fn tick_theme(&mut self, ctx: &egui::Context) {
        let target = Theme::from_seed(self.seed, self.theme_dark);
        if let Some((from, t)) = &mut self.theme_anim {
            let dt = ctx.input(|i| i.stable_dt).clamp(0.0, 0.05);
            *t = (*t + dt / 0.32).min(1.0);
            let e = 1.0 - (1.0 - *t).powi(3);
            self.theme = Theme {
                dark: self.theme_dark,
                pal: crate::theme::lerp_palette(from, &target.pal, e),
            };
            if *t >= 1.0 {
                self.theme_anim = None;
                self.theme = target;
            }
            ctx.request_repaint();
        } else {
            self.theme = target;
        }
        self.theme.apply(ctx);
    }

    fn apply_training_command(&mut self) {
        let (applied, extra, problems) = apply_flags(&mut self.form, &self.cmd_input);
        if applied == 0 && extra == 0 {
            self.cmd_status = if problems.is_empty() {
                tr("Не нашёл флагов — нужна строка вида --model ...").into()
            } else {
                crate::tf!("Не понял: {}", "Did not understand: {}", problems.join(", "))
            };
            return;
        }
        self.plan_applied = true;
        self.checked_sig = None;
        let mut status = crate::tf!("Применено: {applied} флагов", "Applied: {applied} flags");
        if extra > 0 {
            status.push_str(&crate::tf!(" · в доп. аргументы: {extra}", " · into extra args: {extra}"));
        }
        if !problems.is_empty() {
            status.push_str(&crate::tf!(" · пропущено: {}", " · skipped: {}", problems.join(", ")));
        }
        self.cmd_status = status;
        self.start_check();
    }

    fn start_check(&mut self) {
        let settings = self.form.settings();
        self.checked_sig = Some(settings.clone());
        let (tx, rx) = std::sync::mpsc::channel();
        let python = self.base.python.clone();
        let engine = self.base.engine.clone();
        let ws = self.base.ws.clone();
        std::thread::spawn(move || {
            let _ = tx.send(validate::run(&python, &engine, &ws, &settings));
        });
        self.check = CheckState::Running(rx);
    }

    fn apply_plan(&mut self, r: &Report) {
        let p = &r.plan;
        if let Some(v) = p.max_length {
            self.form.max_length = v.to_string();
        }
        if let Some(v) = p.batch_size {
            self.form.batch_size = v.to_string();
        }
        if let Some(v) = p.grad_accum {
            self.form.grad_accum = v.to_string();
        }
        if let Some(v) = &p.optimizer {
            self.form.optimizer = v.clone();
        }
        if let Some(v) = p.lisa_layers {
            self.form.lisa_layers = v.to_string();
        }
        if let Some(v) = p.lisa_interval {
            self.form.lisa_interval = v.to_string();
        }
        if let Some(v) = p.lora_r {
            self.form.lora_r = v.to_string();
        }
        if let Some(v) = p.lora_alpha {
            self.form.lora_alpha = v.to_string();
        }
        if let Some(v) = p.save_steps {
            self.form.save_steps = v.to_string();
        }
        if let Some(v) = &p.amp_dtype {
            self.form.amp_dtype = v.clone();
        }
        if let Some(v) = p.gradient_checkpointing {
            self.form.grad_checkpointing = v;
        }
        if let Some(v) = &p.dtype {
            if v != "auto" {
                self.form.dtype = v.clone();
            }
        }
        if let Some(v) = &p.quant4 {
            self.form.quant4 = v.clone();
        }
        if let Some(v) = p.amp {
            self.form.amp = v;
        }
        self.checked_sig = Some(self.form.settings());
    }

    fn poll_check(&mut self) {
        let incoming = match &self.check {
            CheckState::Running(rx) => match rx.try_recv() {
                Ok(res) => Some(res),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(Err(tr("проверка прервалась").to_string())),
            },
            _ => None,
        };
        let Some(res) = incoming else { return };
        match res {
            Ok(report) => {
                if !self.plan_applied {
                    self.apply_plan(&report);
                    self.plan_applied = true;
                }
                let ok = report.errors == 0;
                self.check = CheckState::Done(report);
                if self.pending_start {
                    self.pending_start = false;
                    if ok {
                        self.start_training();
                    } else {
                        self.error =
                            Some(tr("Есть ошибки в настройках — исправь их перед запуском").into());
                    }
                }
            }
            Err(e) => {
                self.check = CheckState::Failed(e.clone());
                if self.pending_start {
                    self.pending_start = false;
                    self.error = Some(crate::tf!("Проверка недоступна: {e}", "Check unavailable: {e}"));
                    self.start_training();
                }
            }
        }
    }

    fn start_clicked(&mut self) {
        let settings = self.form.settings();
        if let CheckState::Done(r) = &self.check {
            if self.checked_sig.as_ref() == Some(&settings) {
                if r.errors == 0 {
                    self.start_training();
                } else {
                    self.error =
                        Some(tr("Есть ошибки в настройках — исправь их перед запуском").into());
                }
                return;
            }
        }
        // Настройки изменились — сначала перепроверим, потом запустим.
        self.pending_start = true;
        self.error = None;
        self.start_check();
    }

    fn start_training(&mut self) {
        let launch = self.form.launch(&self.base);
        match launch.spawn() {
            Ok(runner) => {
                self.runner = Some(runner);
                self.stats = Stats::default();
                self.log.clear();
                self.log.push_back(format!("$ {}", launch.cmdline()));
                self.paused = false;
                self.quitting = false;
                self.exit_code = None;
                self.error = None;
                self.phase = Phase::Running;
            }
            Err(e) => {
                self.error = Some(crate::tf!(
                    "Не удалось запустить {} ({}). Укажи интерпретатор: --python <путь к python.exe>",
                    "Failed to start {} ({}). Specify the interpreter: --python <path to python.exe>",
                    launch.python, e
                ))
            }
        }
    }

    fn push_line(&mut self, line: String) {
        let was_step = self.stats.feed(&line);
        if was_step && self.stats.started.is_none() {
            self.stats.begin();
        }
        if !was_step {
            self.log.push_back(line);
            while self.log.len() > LOG_LIMIT {
                self.log.pop_front();
            }
        }
    }

    fn send(&mut self, cmd: &str) {
        if let Some(r) = self.runner.as_mut() {
            let _ = r.send(cmd);
            self.log.push_back(format!("> {cmd}"));
        }
    }

    fn poll(&mut self) {
        let mut lines = Vec::new();
        let mut exit = None;
        if let Some(r) = self.runner.as_mut() {
            lines = r.drain();
            exit = r.poll_exit();
        }
        for line in lines {
            self.push_line(line);
        }
        if let Some(code) = exit {
            self.exit_code = Some(code);
            self.phase = Phase::Done;
        }
    }
}

impl eframe::App for GuiApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.chat.poll_scan(&self.base.ws);
        self.chat.poll_params();
        if !self.splash_fading
            && !self.chat.is_scanning()
            && std::time::Instant::now() >= self.splash_until
        {
            self.splash_fading = true;
        }
        if self.splash_fading {
            let dt = ctx.input(|i| i.stable_dt).clamp(0.0, 0.05);
            self.splash_alpha = (self.splash_alpha - dt / 0.28).max(0.0);
            if self.splash_alpha <= 0.0 {
                self.initializing = false;
            }
            ctx.request_repaint();
        } else if self.initializing {
            ctx.request_repaint_after(std::time::Duration::from_millis(30));
        }
        self.chat.poll();
        self.downloads.poll(&self.base.python, &self.base.engine, &self.base.ws);
        if self.chat.is_busy() || self.downloads.busy {
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }
        if matches!(self.check, CheckState::Running(_)) {
            self.poll_check();
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }
        if self.phase == Phase::Running {
            self.poll();
            ctx.request_repaint_after(std::time::Duration::from_millis(120));
        }

        // Тема должна быть активна в каждом кадре: иначе стандартные
        // виджеты (ComboBox, TextEdit) рисуются дефолтным светлым стилем.
        self.tick_theme(ctx);
        let p = self.theme.pal;

        draw_resize_edges(ctx);
        self.draw_titlebar(ctx);

        egui::SidePanel::left("nav")
            .exact_width(264.0)
            .frame(
                egui::Frame::none()
                    .fill(p.surface_container_low)
                    .inner_margin(egui::Margin::symmetric(12.0, 8.0)),
            )
            .show(ctx, |ui| {
                ui.add_space(4.0);
                if new_chat_button(ui, &self.theme).clicked() {
                    self.chat.new_chat();
                    self.tab = Tab::Chat;
                }
                ui.add_space(6.0);
                for tab in [Tab::Chat, Tab::Download, Tab::Train, Tab::Settings] {
                    let selected = self.tab == tab;
                    if nav_item(ui, &self.theme, tab, tab_label(tab), selected).clicked() {
                        if self.tab != tab {
                            self.scroll_top = true;
                            self.content_alpha = 0.0;
                        }
                        self.tab = tab;
                        if tab == Tab::Train && self.checked_sig.is_none() {
                            self.start_check();
                        }
                    }
                }

                ui.add_space(14.0);
                self.theme.hairline(ui);
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(tr("ПРОЕКТЫ"))
                            .font(egui::FontId::new(10.0, egui::FontFamily::Monospace))
                            .color(p.on_surface_variant),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if add_icon_button(ui, &self.theme).clicked() {
                            let start = if self.settings.active_project.trim().is_empty() {
                                self.base.ws.display().to_string()
                            } else {
                                self.settings.active_project.clone()
                            };
                            self.folder_pick = Some(FolderPicker::new(Some(&start)));
                        }
                    });
                });
                ui.add_space(2.0);
                if self.settings.projects.is_empty() {
                    ui.label(self.theme.label(tr("Нет проектов — нажми +")));
                } else {
                    let projects = self.settings.projects.clone();
                    for path in projects {
                        let active = self.settings.active_project == path;
                        let (open, remove) = project_row(ui, &self.theme, &path, active);
                        if remove {
                            self.remove_project(&path);
                        } else if open {
                            self.set_project(path);
                        }
                    }
                }
                ui.add_space(14.0);
                section_label(ui, &self.theme, tr("Недавние"));
                ui.add_space(2.0);
                let recents: Vec<String> = self
                    .chat
                    .messages
                    .iter()
                    .filter(|m| m.role == "user")
                    .map(|m| m.content.split_whitespace().collect::<Vec<_>>().join(" "))
                    .collect();
                if recents.is_empty() {
                    ui.label(self.theme.label(tr("Нет недавних чатов")));
                } else {
                    for text in recents.iter().rev().take(6) {
                        let short: String = text.chars().take(34).collect();
                        if ui
                            .add(
                                egui::Button::new(
                                    RichText::new(short).size(13.0).color(p.on_surface_variant),
                                )
                                .fill(egui::Color32::TRANSPARENT)
                                .stroke(egui::Stroke::NONE)
                                .rounding(egui::Rounding::same(4.0)),
                            )
                            .clicked()
                        {
                            self.chat.input = text.clone();
                            self.tab = Tab::Chat;
                        }
                    }
                }

                ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        let (mark, _) = ui.allocate_exact_size(
                            egui::vec2(7.0, 7.0),
                            egui::Sense::hover(),
                        );
                        ui.painter().rect_filled(mark, egui::Rounding::ZERO, p.primary);
                        ui.add_space(6.0);
                        ui.vertical(|ui| {
                            ui.label(
                                RichText::new("LOCAL FORGE")
                                    .font(egui::FontId::new(10.0, egui::FontFamily::Monospace))
                                    .color(p.on_surface),
                            );
                            ui.label(
                                RichText::new(tr("работает локально"))
                                    .font(egui::FontId::new(9.5, egui::FontFamily::Monospace))
                                    .color(p.on_surface_variant),
                            );
                        });
                    });
                    ui.add_space(8.0);
                    self.theme.hairline(ui);
                });
            });

        if self.tab == Tab::Train {
        egui::SidePanel::right("preview")
            .exact_width(320.0)
            .frame(
                egui::Frame::none()
                    .fill(p.surface)
                    .inner_margin(egui::Margin::symmetric(12.0, 8.0)),
            )
            .show(ctx, |ui| self.ui_preview(ui));

        egui::TopBottomPanel::bottom("actions")
            .exact_height(76.0)
            .frame(
                egui::Frame::none()
                    .fill(p.surface_container)
                    .inner_margin(egui::Margin::symmetric(24.0, 10.0)),
            )
            .show(ctx, |ui| {
                ui.horizontal(|ui| self.ui_actions(ui));
            });
        }

        egui::CentralPanel::default()
            .frame(
                egui::Frame::none()
                    .fill(p.surface)
                    .inner_margin(egui::Margin::symmetric(24.0, 14.0)),
            )
            .show(ctx, |ui| {
                ui.multiply_opacity(self.content_alpha);
                ui.add_space(4.0);
                if self.tab == Tab::Chat {
                    // Чат сам держит композер у нижнего края и скроллит
                    // только ленту сообщений.
                    if let Some(action) = self.chat.draw(
                        ui,
                        &self.theme,
                        &self.base.python,
                        &self.base.engine,
                        &self.base.ws,
                    ) {
                        if action == crate::chat::ChatAction::OpenHub {
                            self.tab = Tab::Download;
                        }
                    }
                    // MoE-режим можно переключить и в «Параметрах» чата — сохраняем.
                    if self.settings.chat_moe_lazy != self.chat.moe_lazy {
                        self.settings.chat_moe_lazy = self.chat.moe_lazy;
                        self.persist_settings();
                    }
                } else if self.tab == Tab::Download {
                    // Хаб сам держит шапку и две независимые прокрутки.
                    self.ui_hub(ui);
                } else {
                ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        match self.tab {
                            Tab::Chat | Tab::Download => {}
                            Tab::Settings => self.ui_settings(ui),
                            Tab::Train => match self.phase {
                                Phase::Setup => self.ui_setup(ui),
                                Phase::Running => self.ui_running(ui),
                                Phase::Done => self.ui_done(ui),
                            },
                        }
                        ui.add_space(16.0);
                    });
                }
            });

        if self.content_alpha < 1.0 {
            let dt = ctx.input(|i| i.stable_dt).clamp(0.0, 0.05);
            self.content_alpha = (self.content_alpha + dt / 0.18).min(1.0);
            ctx.request_repaint();
        }

        let theme_now = self.theme;
        let picker_result = match &mut self.folder_pick {
            Some(picker) => Some(picker.show(ctx, &theme_now)),
            None => None,
        };
        match picker_result {
            Some(PickerResult::Picked(path)) => {
                self.folder_pick = None;
                self.add_project(path.display().to_string());
            }
            Some(PickerResult::Cancelled) => self.folder_pick = None,
            Some(PickerResult::Open) | None => {}
        }

        self.sync_chat_settings();

        if self.splash_alpha > 0.001 {
            self.draw_splash(ctx);
        }
    }
}

/// Пункт M3 navigation drawer: индикатор-пилюля secondary_container.
fn nav_item(ui: &mut egui::Ui, theme: &Theme, tab: Tab, label: &str, selected: bool) -> egui::Response {
    let p = theme.pal;
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 44.0),
        egui::Sense::click(),
    );
    let painter = ui.painter();
    if resp.hovered() && !selected {
        painter.rect_filled(rect, egui::Rounding::same(4.0), p.surface_container);
    }
    if selected {
        let bar = egui::Rect::from_min_size(
            egui::pos2(rect.left(), rect.top() + 2.0),
            egui::vec2(3.0, rect.height() - 4.0),
        );
        painter.rect_filled(bar, egui::Rounding::ZERO, p.primary);
    }
    let fg = if selected { p.on_surface } else { p.on_surface_variant };
    nav_icon(painter, rect, tab, fg, fg);
    painter.text(
        egui::pos2(rect.left() + 48.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        theme::title_font(14.0),
        fg,
    );
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Иконка «+» для добавления проекта.
fn add_icon_button(ui: &mut egui::Ui, theme: &Theme) -> egui::Response {
    let p = theme.pal;
    let (rect, resp) = ui.allocate_exact_size(egui::vec2(24.0, 24.0), egui::Sense::click());
    if resp.hovered() {
        ui.painter()
            .circle_filled(rect.center(), 12.0, p.surface_container_high);
    }
    theme::draw_icon(
        ui.painter(),
        rect.center(),
        theme::icons::ADD,
        18.0,
        p.on_surface_variant,
    );
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Строка проекта: папка, имя, удаление по наведению.
/// Возвращает (открыть, удалить).
fn project_row(
    ui: &mut egui::Ui,
    theme: &Theme,
    path: &str,
    active: bool,
) -> (bool, bool) {
    let p = theme.pal;
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 36.0),
        egui::Sense::click(),
    );
    let painter = ui.painter().clone();
    if active {
        painter.rect_filled(rect, egui::Rounding::same(4.0), p.surface_container);
        let bar = egui::Rect::from_min_size(
            egui::pos2(rect.left() + 4.0, rect.center().y - 12.0),
            egui::vec2(3.0, 24.0),
        );
        painter.rect_filled(bar, egui::Rounding::ZERO, p.primary);
    } else if resp.hovered() {
        painter.rect_filled(rect, egui::Rounding::same(4.0), p.surface_container);
    }
    let fg = p.on_surface;
    theme::draw_icon(
        &painter,
        egui::pos2(rect.left() + 22.0, rect.center().y),
        theme::icons::FOLDER,
        17.0,
        fg,
    );
    let name = std::path::Path::new(path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string());
    let max_chars = (((rect.width() - 72.0) / 7.2).max(4.0)) as usize;
    let shown: String = if name.chars().count() > max_chars {
        format!("{}…", name.chars().take(max_chars.saturating_sub(1)).collect::<String>())
    } else {
        name
    };
    painter.text(
        egui::pos2(rect.left() + 38.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        shown,
        theme::title_font(13.0),
        fg,
    );

    let del_rect = egui::Rect::from_center_size(
        egui::pos2(rect.right() - 15.0, rect.center().y),
        egui::vec2(24.0, 24.0),
    );
    let del = ui.interact(
        del_rect,
        ui.id().with(("project_del", path)),
        egui::Sense::click(),
    );
    let show_del = resp.hovered() || del.hovered() || active;
    if show_del {
        if del.hovered() {
            painter.circle_filled(del_rect.center(), 11.0, p.error_container);
        }
        let c = del_rect.center();
        let stroke = egui::Stroke::new(
            1.3_f32,
            if del.hovered() { p.on_error_container } else { p.on_surface_variant },
        );
        painter.line_segment(
            [egui::pos2(c.x - 3.5, c.y - 3.5), egui::pos2(c.x + 3.5, c.y + 3.5)],
            stroke,
        );
        painter.line_segment(
            [egui::pos2(c.x - 3.5, c.y + 3.5), egui::pos2(c.x + 3.5, c.y - 3.5)],
            stroke,
        );
    }
    let row_clicked = resp.clicked();
    let del_clicked = del.clicked();
    resp.on_hover_text(path);
    (row_clicked && !del_clicked, del_clicked)
}

/// M3 extended FAB в начале drawer: «Новый чат».
fn new_chat_button(ui: &mut egui::Ui, theme: &Theme) -> egui::Response {
    let p = theme.pal;
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 44.0),
        egui::Sense::click(),
    );
    let fill = if resp.hovered() {
        theme::lerp_color(p.primary, p.on_primary, 0.08)
    } else {
        p.primary
    };
    let painter = ui.painter();
    painter.rect_filled(rect, egui::Rounding::same(10.0), fill);
    let cy = rect.center().y;
    theme::draw_icon(
        painter,
        egui::pos2(rect.left() + 22.0, cy),
        theme::icons::ADD,
        19.0,
        p.on_primary,
    );
    painter.text(
        egui::pos2(rect.left() + 42.0, cy),
        egui::Align2::LEFT_CENTER,
        tr("Новый чат"),
        theme::title_font(14.0),
        p.on_primary,
    );
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

fn page_header(ui: &mut egui::Ui, theme: &Theme, overline: &str, title: &str, subtitle: &str) {
    ui.add_space(2.0);
    ui.label(theme.overline(overline));
    ui.add_space(2.0);
    ui.label(theme.display(title, 36.0));
    if !subtitle.is_empty() {
        ui.add_space(2.0);
        ui.label(theme.label(subtitle));
    }
    ui.add_space(10.0);
    theme.hairline(ui);
    ui.add_space(12.0);
}

fn section_label(ui: &mut egui::Ui, theme: &Theme, text: &str) {
    ui.add_space(6.0);
    theme.hairline(ui);
    ui.add_space(6.0);
    ui.label(
        RichText::new(text.to_uppercase())
            .font(egui::FontId::new(10.0, egui::FontFamily::Monospace))
            .color(theme.pal.on_surface_variant),
    );
}

#[derive(Clone, Copy)]
enum WindowButton {
    Minimize,
    Maximize(bool),
    Close,
}

/// Кнопка окна в шапке: 40×32, при наведении — M3-круг.
fn window_button(ui: &mut egui::Ui, theme: &Theme, kind: WindowButton) -> egui::Response {
    let p = theme.pal;
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(40.0, 32.0), egui::Sense::click());
    let hover = resp.hovered();
    let close = matches!(kind, WindowButton::Close);
    let painter = ui.painter();
    if hover {
        let fill = if close {
            p.error_container
        } else {
            p.surface_container_highest
        };
        painter.circle_filled(rect.center(), 15.0, fill);
    }
    let color = if close && hover {
        p.on_error_container
    } else {
        p.on_surface_variant
    };
    let stroke = egui::Stroke::new(1.4_f32, color);
    let c = rect.center();
    match kind {
        WindowButton::Minimize => {
            theme::draw_icon(painter, c, theme::icons::MINIMIZE, 16.0, color);
        }
        WindowButton::Maximize(false) => {
            theme::draw_icon(painter, c, theme::icons::MAXIMIZE, 14.0, color);
        }
        WindowButton::Maximize(true) => {
            let back = egui::Rect::from_min_size(egui::pos2(c.x - 4.0, c.y - 5.5), egui::vec2(8.5, 8.5));
            let front = egui::Rect::from_min_size(egui::pos2(c.x - 5.5, c.y - 4.0), egui::vec2(8.5, 8.5));
            painter.rect_stroke(back, egui::Rounding::same(2.0), stroke);
            painter.rect_filled(back.intersect(front), 0.0, p.surface_container_low);
            painter.rect_stroke(front, egui::Rounding::same(2.0), stroke);
        }
        WindowButton::Close => {
            theme::draw_icon(painter, c, theme::icons::CLOSE, 16.0, color);
        }
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Края окна без нативной рамки: курсор-стрелки и старт ресайза.
fn draw_resize_edges(ctx: &egui::Context) {
    let maximized = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
    if maximized {
        return;
    }
    let Some(pos) = ctx.input(|i| i.pointer.hover_pos()) else {
        return;
    };
    let r = ctx.screen_rect();
    let edge = 6.0;
    let corner = 12.0;
    let (l, rr) = (pos.x - r.left(), r.right() - pos.x);
    let (t, b) = (pos.y - r.top(), r.bottom() - pos.y);
    use egui::viewport::ResizeDirection as D;
    let dir = if l < corner && t < corner {
        Some((D::NorthWest, egui::CursorIcon::ResizeNwSe))
    } else if rr < corner && t < corner {
        Some((D::NorthEast, egui::CursorIcon::ResizeNeSw))
    } else if l < corner && b < corner {
        Some((D::SouthWest, egui::CursorIcon::ResizeNeSw))
    } else if rr < corner && b < corner {
        Some((D::SouthEast, egui::CursorIcon::ResizeNwSe))
    } else if l < edge {
        Some((D::West, egui::CursorIcon::ResizeHorizontal))
    } else if rr < edge {
        Some((D::East, egui::CursorIcon::ResizeHorizontal))
    } else if t < edge {
        Some((D::North, egui::CursorIcon::ResizeVertical))
    } else if b < edge {
        Some((D::South, egui::CursorIcon::ResizeVertical))
    } else {
        None
    };
    if let Some((dir, icon)) = dir {
        ctx.set_cursor_icon(icon);
        if ctx.input(|i| i.pointer.primary_pressed()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::BeginResize(dir));
        }
    }
}

/// Простые векторные иконки, чтобы не тянуть icon-шрифт.
fn nav_icon(painter: &egui::Painter, rect: egui::Rect, tab: Tab, color: egui::Color32, _accent: egui::Color32) {
    let code = match tab {
        Tab::Chat => theme::icons::CHAT,
        Tab::Download => theme::icons::DASHBOARD,
        Tab::Train => theme::icons::TUNE,
        Tab::Settings => theme::icons::SETTINGS,
    };
    theme::draw_icon(
        painter,
        egui::pos2(rect.left() + 22.0, rect.center().y),
        code,
        21.0,
        color,
    );
}

impl GuiApp {
    fn preview_row(&self, ui: &mut egui::Ui, label: &str, value: String) {
        let p = self.theme.pal;
        ui.horizontal(|ui| {
            ui.label(RichText::new(label).size(12.5).color(p.on_surface_variant));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(value).size(12.5).color(p.on_surface));
            });
        });
    }

    fn ui_hub(&mut self, ui: &mut egui::Ui) {
        // CPU/RAM берём из системы — не ждём Python-проверку.
        let (ram, total) = crate::sysinfo::ram();
        let vram = match &self.check {
            CheckState::Done(rep) => rep.hardware.vram_gb.unwrap_or(0.0),
            _ => 0.0,
        };
        let cpu = crate::sysinfo::cpu_count() as u32;
        let action = self.downloads.draw(
            ui,
            &self.theme,
            &self.base.python,
            &self.base.engine,
            &self.base.ws,
            &self.chat.models,
            cpu,
            ram,
            total,
            vram,
            &self.settings.hf_user,
            &self.settings.hub_quants,
        );
        match action {
            Some(crate::downloads::HubAction::UseModel(path)) => {
                self.chat.select_model(path);
                self.tab = Tab::Chat;
                self.content_alpha = 0.0;
            }
            Some(crate::downloads::HubAction::Downloaded) => {
                self.chat.rescan(&self.base.ws);
            }
            None => {}
        }
    }

    fn ui_settings(&mut self, ui: &mut egui::Ui) {
        let p = self.theme.pal;
        let mut dirty = false;
        ui.set_max_width(1040.0);
        page_header(
            ui,
            &self.theme,
            tr("Конфигурация"),
            tr("Настройки"),
            tr("Внешний вид, чат, обучение и пути"),
        );

        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.section(tr("Внешний вид")));
            ui.horizontal(|ui| {
                ui.label(self.theme.label(tr("Тема")));
                let cur = if self.theme_dark { tr("Тёмная") } else { tr("Светлая") };
                if let Some(v) = theme::segmented(ui, &self.theme, &[tr("Светлая"), tr("Тёмная")], cur) {
                    let dark = v == tr("Тёмная");
                    if dark != self.theme_dark {
                        let from = self.theme.pal;
                        self.theme_dark = dark;
                        self.theme_anim = Some((from, 0.0));
                        dirty = true;
                    }
                }
            });
            ui.horizontal(|ui| {
                ui.label(self.theme.label(tr("Язык интерфейса")));
                let cur = if self.settings.lang == "en" { "EN" } else { "RU" };
                if let Some(v) = theme::segmented(ui, &self.theme, &["RU", "EN"], cur) {
                    let lang = if v == "EN" { "en" } else { "ru" };
                    if lang != self.settings.lang {
                        self.settings.lang = lang.into();
                        crate::i18n::set_lang(&self.settings.lang);
                        dirty = true;
                    }
                }
            });
            ui.add_space(4.0);
            ui.horizontal_wrapped(|ui| {
                ui.label(self.theme.label(tr("Цвет схемы")));
                if theme::button(ui, &self.theme, tr("Системный"), ButtonKind::Tonal, true).clicked() {
                    let from = self.theme.pal;
                    self.settings.seed = "system".into();
                    let new_seed = crate::settings::resolve_seed("system");
                    if new_seed != self.seed {
                        self.seed = new_seed;
                        self.theme_anim = Some((from, 0.0));
                    }
                    dirty = true;
                }
                for hex in ["#0e9f6e", "#006fc7", "#7c4dff", "#f57c00", "#d32f2f", "#c2185b"] {
                    let rgb = crate::settings::parse_seed(hex).unwrap_or([0, 0, 0]);
                    let selected = self.seed == rgb;
                    let swatch = egui::Button::new("")
                        .fill(egui::Color32::from_rgb(rgb[0], rgb[1], rgb[2]))
                        .rounding(egui::Rounding::same(9.0))
                        .min_size(egui::vec2(26.0, 26.0))
                        .stroke(if selected {
                            egui::Stroke::new(2.0_f32, p.on_surface)
                        } else {
                            egui::Stroke::NONE
                        });
                    if ui.add(swatch).clicked() {
                        let from = self.theme.pal;
                        self.seed = rgb;
                        self.settings.seed = hex.to_string();
                        self.theme_anim = Some((from, 0.0));
                        dirty = true;
                    }
                }
                let mut custom = self.seed;
                if ui.color_edit_button_srgb(&mut custom).changed() {
                    let from = self.theme.pal;
                    self.seed = custom;
                    self.settings.seed = crate::settings::seed_hex(custom);
                    self.theme_anim = Some((from, 0.0));
                    dirty = true;
                }
            });
        });

        ui.add_space(10.0);
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.section(tr("Чат")));
            ui.horizontal_wrapped(|ui| {
                ui.label(self.theme.label(tr("Потоки")));
                if ui.add(egui::DragValue::new(&mut self.settings.chat_threads).range(1..=64)).changed() {
                    dirty = true;
                }
                ui.add_space(10.0);
                ui.label(self.theme.label(tr("Контекст")));
                if ui.add(egui::DragValue::new(&mut self.settings.chat_ctx).range(256..=32768)).changed() {
                    dirty = true;
                }
                ui.add_space(10.0);
                ui.label(self.theme.label(tr("Макс. токенов")));
                if ui.add(egui::DragValue::new(&mut self.settings.chat_max_tokens).range(16..=4096)).changed() {
                    dirty = true;
                }
                ui.add_space(10.0);
                ui.label(self.theme.label("temp"));
                if ui.add(egui::DragValue::new(&mut self.settings.chat_temp).range(0.0..=2.0).speed(0.01)).changed() {
                    dirty = true;
                }
                ui.add_space(10.0);
                ui.label(self.theme.label(tr("KV-кэш")));
                let cur = self.settings.chat_kv.clone();
                egui::ComboBox::from_id_salt("set_kv").selected_text(cur).show_ui(ui, |ui| {
                    for kv in ["f16", "q8_0", "q4_0"] {
                        if ui.selectable_value(&mut self.settings.chat_kv, kv.to_string(), kv).changed() {
                            dirty = true;
                        }
                    }
                });
            });
            ui.add_space(8.0);
            if ui
                .checkbox(
                    &mut self.settings.chat_moe_lazy,
                    tr("MoE: грузить только используемые эксперты (эксперимент)"),
                )
                .changed()
            {
                dirty = true;
            }
            if self.settings.chat_moe_lazy {
                ui.label(self.theme.label(tr(
                    "Экспериментально: эксперты читаются с SSD по требованию. Сильно нагружает диск и CPU. Только для transformers-моделей (не GGUF).",
                )));
            }
        });

        ui.add_space(10.0);
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.section(tr("Хаб моделей")));
            text_row(
                ui,
                &self.theme,
                tr("Автор релизов на Hugging Face"),
                &mut self.settings.hf_user,
                "YoSaNoo",
            );
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(self.theme.label(tr("Кванты в ленте")));
                let cur = if self.settings.hub_quants == "runnable" {
                    tr("Под железо")
                } else {
                    "Unsloth"
                };
                if let Some(v) =
                    theme::segmented(ui, &self.theme, &["Unsloth", tr("Под железо")], cur)
                {
                    let mode = if v == tr("Под железо") { "runnable" } else { "unsloth" };
                    if mode != self.settings.hub_quants {
                        self.settings.hub_quants = mode.to_string();
                        dirty = true;
                    }
                }
            });
        });

        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.section(tr("Обучение по умолчанию")));
            ui.horizontal(|ui| {
                ui.label(self.theme.label(tr("Профиль нагрузки")));
                if let Some(v) = theme::segmented(
                    ui,
                    &self.theme,
                    &["game", "work", "max", "turbo"],
                    &self.settings.train_preset.clone(),
                ) {
                    self.settings.train_preset = v.clone();
                    self.form.preset = v;
                    dirty = true;
                }
            });
        });

        ui.add_space(10.0);
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.section(tr("Пути")));
            ui.horizontal(|ui| {
                ui.label(self.theme.label(tr("Python (пусто — авто)")));
                if ui
                    .add(
                        egui::TextEdit::singleline(&mut self.settings.python)
                            .desired_width(320.0)
                            .hint_text("python.exe"),
                    )
                    .changed()
                {
                    dirty = true;
                }
            });
            ui.horizontal(|ui| {
                ui.label(self.theme.label(tr("Движок (пусто — авто)")));
                if ui
                    .add(
                        egui::TextEdit::singleline(&mut self.settings.engine)
                            .desired_width(320.0)
                            .hint_text(tr("папка с main.py")),
                    )
                    .changed()
                {
                    dirty = true;
                }
            });
            ui.label(
                self.theme
                    .label(crate::tf!("Рабочая папка: {}", "Working folder: {}", self.base.ws.display())),
            );
        });

        ui.add_space(10.0);
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            if theme::button(ui, &self.theme, tr("Сбросить настройки"), ButtonKind::Tonal, true).clicked() {
                self.settings = crate::settings::AppSettings::default();
                self.theme_dark = false;
                self.seed = crate::settings::resolve_seed("system");
                self.theme = Theme::from_seed(self.seed, false);
                self.theme_anim = None;
                self.apply_settings();
                self.persist_settings();
            }
            ui.label(self.theme.label(tr("Тема, цвет, параметры чата и пути.")));
        });

        ui.add_space(10.0);
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.section(tr("О программе")));
            ui.label(self.theme.label(crate::tf!(
                "Local Forge {} · публичная бета",
                "Local Forge {} · public beta",
                env!("CARGO_PKG_VERSION")
            )));
            ui.label(self.theme.label(tr("Локальный запуск GGUF-моделей, хаб, обучение и публичный API. Без облака.")));
            ui.label(self.theme.label(tr("Бета: возможны ошибки. Публичный API — только через Radmin VPN, ключ фиксированный.")));
        });

        if dirty {
            self.apply_settings();
            self.persist_settings();
        }
    }

    fn ui_preview(&mut self, ui: &mut egui::Ui) {
        let p = self.theme.pal;
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            let (status_text, status_color) = match &self.check {
                CheckState::Done(rep) if rep.errors > 0 => {
                    (crate::tf!("ошибок: {}", "errors: {}", rep.errors), p.error)
                }
                CheckState::Done(rep) if rep.warnings > 0 => {
                    (crate::tf!("предупреждений: {}", "warnings: {}", rep.warnings), warn_color())
                }
                CheckState::Done(_) => (tr("проверка пройдена").to_string(), ok_color()),
                CheckState::Running(_) => (tr("проверяю…").to_string(), p.primary),
                _ => (tr("не проверено").to_string(), p.on_surface_variant),
            };
            ui.label(
                RichText::new(tr("Предпросмотр запуска"))
                    .font(egui::FontId::new(18.0, theme::bold_family()))
                    .color(p.on_surface),
            );
            ui.label(RichText::new(status_text).size(12.0).color(status_color));
            ui.add_space(8.0);
            let cfg = self.form.clone();
            self.preview_row(ui, tr("Модель"), cfg.model.clone());
            self.preview_row(ui, tr("Датасет"), cfg.train_file.clone());
            self.preview_row(ui, tr("Метод"), format!("{}", cfg.tune_mode));
            self.preview_row(ui, "LoRA r", if cfg.lora_r.is_empty() { tr("авто").into() } else { cfg.lora_r.clone() });
            self.preview_row(ui, tr("Контекст"), if cfg.max_length.is_empty() { tr("авто").into() } else { cfg.max_length.clone() });
            self.preview_row(ui, "Batch", if cfg.batch_size.is_empty() { tr("авто").into() } else { cfg.batch_size.clone() });
            self.preview_row(ui, "Grad accum", if cfg.grad_accum.is_empty() { tr("авто").into() } else { cfg.grad_accum.clone() });
            self.preview_row(ui, "LR", cfg.lr.clone());
            self.preview_row(ui, tr("Оптимизатор"), cfg.optimizer.clone());
            if let CheckState::Done(rep) = &self.check {
                let hw = &rep.hardware;
                self.preview_row(ui, tr("Оборудование"), if hw.cuda.unwrap_or(false) {
                    crate::tf!("{} / {:.1} ГБ VRAM", "{} / {:.1} GB VRAM", hw.gpu.clone().unwrap_or_default(), hw.vram_gb.unwrap_or(0.0))
                } else {
                    crate::tf!("CPU {} потоков", "CPU {} threads", hw.cpu.unwrap_or(0))
                });
                self.preview_row(ui, tr("RAM свободно"), crate::tf!("{:.1} ГБ", "{:.1} GB", hw.ram_avail_gb.unwrap_or(0.0)));
            }
        });
    }

    fn ui_actions(&mut self, ui: &mut egui::Ui) {
        let p = self.theme.pal;
        match self.phase {
            Phase::Setup => {
                let checking = matches!(self.check, CheckState::Running(_));
                if theme::button(ui, &self.theme, tr("Проверить"), ButtonKind::Tonal, !checking).clicked() {
                    self.start_check();
                }
                if theme::button(ui, &self.theme, tr("Начать обучение"), ButtonKind::Filled, !checking)
                    .clicked()
                {
                    self.start_clicked();
                }
                if checking {
                    ui.label(self.theme.label(tr("Проверяю настройки…")));
                } else if let CheckState::Done(r) = &self.check {
                    if r.errors > 0 {
                        ui.label(RichText::new(crate::tf!("Ошибок: {}", "Errors: {}", r.errors)).color(p.error).size(13.0));
                    } else if r.warnings > 0 {
                        ui.label(
                            RichText::new(crate::tf!("Предупреждений: {}", "Warnings: {}", r.warnings))
                                .color(warn_color())
                                .size(13.0),
                        );
                    } else {
                        ui.label(RichText::new(tr("Проверка пройдена")).color(p.primary).size(13.0));
                    }
                }
                if let Some(err) = &self.error {
                    ui.label(RichText::new(err).color(p.error).size(13.0));
                }
            }
            Phase::Running => {
                let profile = self.stats.profile.clone().unwrap_or_else(|| self.form.preset.clone());
                if let Some(picked) =
                    theme::segmented(ui, &self.theme, &["game", "work", "max", "turbo"], &profile)
                {
                    self.send(&picked);
                }
                if self.paused {
                    if theme::button(ui, &self.theme, tr("Продолжить"), ButtonKind::Tonal, !self.quitting).clicked() {
                        self.paused = false;
                        self.send("resume");
                    }
                } else if theme::button(ui, &self.theme, tr("Пауза"), ButtonKind::Tonal, !self.quitting).clicked() {
                    self.paused = true;
                    self.send("pause");
                }
                if theme::button(ui, &self.theme, tr("Сохранить"), ButtonKind::Tonal, !self.quitting).clicked() {
                    self.send("save");
                }
                if theme::button(ui, &self.theme, tr("Стоп"), ButtonKind::Danger, !self.quitting).clicked() {
                    self.quitting = true;
                    self.send("stop");
                }
            }
            Phase::Done => {
                if theme::button(ui, &self.theme, tr("Открыть папку"), ButtonKind::Tonal, true).clicked() {
                    let dir = self.base.ws.join(self.form.out.trim());
                    #[cfg(windows)]
                    {
                        let _ = std::process::Command::new("explorer").arg(&dir).spawn();
                    }
                    #[cfg(target_os = "linux")]
                    {
                        let _ = std::process::Command::new("xdg-open").arg(&dir).spawn();
                    }
                    #[cfg(target_os = "macos")]
                    {
                        let _ = std::process::Command::new("open").arg(&dir).spawn();
                    }
                }
                if theme::button(ui, &self.theme, tr("Ещё раз"), ButtonKind::Filled, true).clicked() {
                    self.phase = Phase::Setup;
                    self.runner = None;
                }
            }
        }
    }

    fn ui_setup(&mut self, ui: &mut egui::Ui) {
        let p = self.theme.pal;
        ui.set_max_width(860.0);
        if self.scroll_top && self.tab == Tab::Train {
            // Форма длинная: при первом показе возвращаем прокрутку наверх.
            ui.scroll_to_cursor(Some(egui::Align::Min));
            self.scroll_top = false;
        }
        page_header(
            ui,
            &self.theme,
            tr("Пайплайн"),
            tr("Обучение"),
            tr("Настройки, проверка под железо и запуск на CPU"),
        );

        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.section(tr("Своя команда")));
            ui.label(self.theme.label(
                tr("Вставь команду main.py — флаги заполнят настройки ниже и запустят проверку."),
            ));
            ui.add_space(4.0);
            egui::Frame::none()
                .fill(p.surface_container_highest)
                .rounding(egui::Rounding::same(4.0))
                .inner_margin(egui::Margin::symmetric(12.0, 8.0))
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    ui.add(
                        egui::TextEdit::multiline(&mut self.cmd_input)
                            .frame(false)
                            .desired_rows(2)
                            .desired_width(f32::INFINITY)
                            .hint_text(
                                "python main.py --model models/gemma_base --epochs 2 --packing",
                            ),
                    );
                });
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                let can = !self.cmd_input.trim().is_empty();
                if theme::button(ui, &self.theme, tr("Применить"), ButtonKind::Filled, can).clicked() {
                    self.apply_training_command();
                }
                if theme::button(
                    ui,
                    &self.theme,
                    tr("Очистить"),
                    ButtonKind::Text,
                    !self.cmd_input.is_empty(),
                )
                .clicked()
                {
                    self.cmd_input.clear();
                    self.cmd_status.clear();
                }
                if !self.cmd_status.is_empty() {
                    ui.label(self.theme.label(&self.cmd_status));
                }
            });
        });
        ui.add_space(12.0);

        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.section(tr("Модель и данные")));
            text_row(ui, &self.theme, tr("Модель (папка с safetensors)"), &mut self.form.model, "models/gemma_base");
            text_row(ui, &self.theme, tr("Обучающий датасет (jsonl)"), &mut self.form.train_file, "database/train_11.jsonl");
            text_row(ui, &self.theme, tr("Валидация (не влияет на обучение)"), &mut self.form.val_file, tr("необязательно"));
            text_row(ui, &self.theme, tr("Куда сохранять"), &mut self.form.out, "output_model");
            text_row(ui, &self.theme, tr("Файл личности (подмешивание)"), &mut self.form.identity_file, tr("необязательно"));
        });

        ui.add_space(12.0);
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.section(tr("Метод")));
            ui.horizontal(|ui| {
                ui.label(self.theme.label(tr("Цель")));
                if let Some(v) = theme::segmented(ui, &self.theme, &["sft", "orpo", "heretic"], &self.form.objective) {
                    self.form.objective = v;
                }
            });
            if self.form.objective == "heretic" {
                pair(ui, &self.theme, "Heretic: trials", &mut self.form.heretic_trials, "3", "Startup trials", &mut self.form.heretic_startup_trials, "1");
                combo_row(ui, &self.theme, tr("Heretic: квантование"), &mut self.form.heretic_quant, &["auto", "none", "bnb4"]);
                ui.label(self.theme.label(tr("Heretic меняет веса через ARA, без градиентного обучения. Обучающий датасет и SFT-поля ниже не используются. Нужна merged HF-модель, а не LoRA-чекпоинт.")));
            }
        });

        if self.form.objective != "heretic" {
        ui.add_space(12.0);
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.section(tr("Параметры обучения")));
            pair(ui, &self.theme, tr("Эпохи"), &mut self.form.epochs, "1", "Learning rate", &mut self.form.lr, "2e-5");
            pair(ui, &self.theme, "Max grad norm", &mut self.form.max_grad_norm, "1.0", tr("Warmup (доля)"), &mut self.form.warmup_ratio, "0.05");
            pair(ui, &self.theme, tr("Контекст (токенов)"), &mut self.form.max_length, tr("авто"), tr("Сэмплов (пусто — все)"), &mut self.form.max_samples, tr("авто"));
            pair(ui, &self.theme, "Batch size", &mut self.form.batch_size, tr("авто"), "Grad accumulation", &mut self.form.grad_accum, tr("авто"));
            pair(ui, &self.theme, tr("Сохранять каждые N шагов"), &mut self.form.save_steps, "50", "LoRA r", &mut self.form.lora_r, "8");
            pair(ui, &self.theme, "LoRA alpha", &mut self.form.lora_alpha, "16", tr("LoRA+ (0 = выкл)"), &mut self.form.lora_plus_ratio, "0");
            ui.horizontal(|ui| {
                ui.label(self.theme.label(tr("Режим весов")));
                if let Some(v) = theme::segmented(ui, &self.theme, &["lora", "full"], &self.form.tune_mode) {
                    self.form.tune_mode = v;
                }
            });
        });

        ui.add_space(12.0);
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.label(tr("ТОЧНОСТЬ, ПАМЯТЬ И ЖЕЛЕЗО")));
            ui.horizontal(|ui| {
                ui.label(self.theme.label(tr("Профиль нагрузки")));
                if let Some(v) = theme::segmented(ui, &self.theme, &["game", "work", "max", "turbo"], &self.form.preset) {
                    self.form.preset = v;
                }
            });
            ui.add_space(4.0);
            combo_row(ui, &self.theme, tr("Оптимизатор"), &mut self.form.optimizer, &["auto", "adamw", "adamw8bit", "adafactor", "sgd", "galore"]);
            combo_row(ui, &self.theme, tr("Точность весов (dtype)"), &mut self.form.dtype, &["auto", "bf16", "fp16", "fp32"]);
            combo_row(ui, &self.theme, "AMP dtype", &mut self.form.amp_dtype, &["auto", "bf16", "fp16"]);
            combo_row(ui, &self.theme, tr("4-битные веса (QLoRA)"), &mut self.form.quant4, &["auto", "on", "off"]);
            ui.checkbox(&mut self.form.amp, tr("AMP — смешанная точность"));
            ui.checkbox(&mut self.form.grad_checkpointing, tr("Gradient checkpointing (меньше RAM, медленнее)"));
            pair(ui, &self.theme, tr("LISA: слоёв (0 = все)"), &mut self.form.lisa_layers, "0", tr("LISA: интервал"), &mut self.form.lisa_interval, "50");
        });

        ui.add_space(12.0);
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.section(tr("Данные и режимы")));
            ui.checkbox(&mut self.form.group_by_length, tr("Length grouping — сортировка по длине (меньше padding)"));
            ui.checkbox(&mut self.form.packing, tr("Packing — склейка диалогов в окно"));
            ui.checkbox(&mut self.form.kd, tr("Дистилляция KD — KL-лосс по top-k шансам учителя"));
            if self.form.kd {
                pair(ui, &self.theme, tr("Вес KD"), &mut self.form.kd_weight, "0.5", tr("Top-k учителя"), &mut self.form.kd_top_k, "8");
                ui.label(self.theme.label(tr("Нужен датасет с полем soft (train_distill.kd.jsonl.gz); packing отключится. Длинные ответы будут пропущены — подними контекст.")));
            }
            ui.checkbox(&mut self.form.cache, tr("Кэш токенизации"));
            pair(ui, &self.theme, tr("Репетиция: файл (anti-forget)"), &mut self.form.replay_file, tr("необязательно"), tr("Доля репетиции"), &mut self.form.replay_ratio, "0.2");
            pair(ui, &self.theme, tr("KL-якорь к базе (0 = выкл)"), &mut self.form.kl_anchor, "0", "ORPO beta", &mut self.form.orpo_beta, "0.1");
            if self.form.objective == "orpo" {
                text_row(ui, &self.theme, tr("Файл пар chosen/rejected"), &mut self.form.orpo_file, "database/orpo.jsonl");
            }
        });
        }

        ui.add_space(12.0);
        egui::CollapsingHeader::new(tr("Дополнительно (экспертные флаги)"))
            .id_salt("advanced")
            .show(ui, |ui| {
                self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    combo_row(ui, &self.theme, tr("Внимание (attn)"), &mut self.form.attn, &["auto", "sdpa", "fa2"]);
                    ui.checkbox(&mut self.form.liger, tr("Liger ядра (Linux+CUDA)"));
                    ui.checkbox(&mut self.form.compile, tr("torch.compile (нужен C++ компилятор)"));
                    text_row(ui, &self.theme, tr("Доп. аргументы main.py (через пробел)"), &mut self.form.extra, "--threads 8 ...");
                });
            });

        ui.add_space(12.0);
        self.ui_checks(ui);
    }

    fn ui_checks(&mut self, ui: &mut egui::Ui) {
        let p = self.theme.pal;
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                ui.label(self.theme.section(tr("Проверка под железо")));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let busy = matches!(self.check, CheckState::Running(_));
                    if theme::button(ui, &self.theme, tr("Перепроверить"), ButtonKind::Outlined, !busy).clicked() {
                        self.start_check();
                    }
                });
            });
            match &self.check {
                CheckState::Idle => {
                    ui.label(self.theme.label(tr("Нажми «Перепроверить», чтобы сверить настройки с железом.")));
                }
                CheckState::Running(_) => {
                    ui.label(self.theme.label(tr("Считаю план и проверяю настройки…")));
                }
                CheckState::Failed(e) => {
                    ui.label(RichText::new(crate::tf!("Проверка не удалась: {e}", "Check failed: {e}")).color(p.error));
                }
                CheckState::Done(r) => {
                    let hw = &r.hardware;
                    let gpu = if hw.cuda.unwrap_or(false) {
                        crate::tf!("{} / {:.1} ГБ VRAM", "{} / {:.1} GB VRAM",
                            hw.gpu.clone().unwrap_or_default(),
                            hw.vram_gb.unwrap_or(0.0)
                        )
                    } else {
                        tr("CPU (без CUDA)").to_string()
                    };
                    ui.label(self.theme.label(crate::tf!("CPU {} | RAM {:.1}/{:.1} ГБ | {} | диск {:.0} ГБ | torch {}", "CPU {} | RAM {:.1}/{:.1} GB | {} | disk {:.0} GB | torch {}",
                        hw.cpu.unwrap_or(0),
                        hw.ram_avail_gb.unwrap_or(0.0),
                        hw.ram_total_gb.unwrap_or(0.0),
                        gpu,
                        hw.disk_free_gb.unwrap_or(0.0),
                        hw.torch.clone().unwrap_or_else(|| "?".into())
                    )))
                    .on_hover_text(format!("python: {}", hw.python.clone().unwrap_or_default()));
                    if let Some(strategy) = &r.plan.strategy {
                        ui.label(self.theme.label(crate::tf!("автоплан: {}, веса {:.1} ГБ, контекст {}, accum {}, оптимизатор {}", "auto plan: {}, weights {:.1} GB, context {}, accum {}, optimizer {}",
                            strategy,
                            r.plan.weights_gb.unwrap_or(0.0),
                            r.plan.max_length.map(|v| v.to_string()).unwrap_or_else(|| "—".into()),
                            r.plan.grad_accum.map(|v| v.to_string()).unwrap_or_else(|| "—".into()),
                            r.plan.optimizer.clone().unwrap_or_else(|| "—".into())
                        )));
                    }
                    ui.add_space(6.0);
                    ScrollArea::vertical().max_height(240.0).show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        for c in r.sorted() {
                            let (mark, color) = match c.level.as_str() {
                                "error" => ("✕", p.error),
                                "warn" => ("!", warn_color()),
                                "ok" => ("✓", ok_color()),
                                _ => ("i", p.on_surface_variant),
                            };
                            ui.horizontal_wrapped(|ui| {
                                ui.label(RichText::new(mark).color(color).strong());
                                ui.label(
                                    RichText::new(format!("[{}] {}", c.field, c.message))
                                        .size(13.0)
                                        .color(color),
                                );
                            });
                        }
                    });
                }
            }
        });
    }

    fn ui_running(&mut self, ui: &mut egui::Ui) {
        let p = self.theme.pal;
        ui.set_max_width(860.0);
        let s = &self.stats;

        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                let (label, color) = if self.quitting {
                    (tr("Останавливаю и сохраняю…"), p.error)
                } else if self.paused {
                    (tr("Пауза"), p.secondary)
                } else {
                    (tr("Обучение идёт"), p.primary)
                };
                ui.label(RichText::new(label).font(egui::FontId::new(15.0, theme::bold_family())).color(color));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let eta = s.eta().map(human_duration).unwrap_or_else(|| "—".into());
                    ui.label(self.theme.label(crate::tf!("осталось {eta}", "{eta} left")));
                });
            });
            ui.add_space(2.0);
            let total = s.total_steps.unwrap_or(0);
            let big = if total > 0 {
                crate::tf!("Шаг {} из {}", "Step {} of {}", s.step, total)
            } else {
                crate::tf!("Шаг {}", "Step {}", s.step)
            };
            ui.label(RichText::new(big).font(egui::FontId::new(32.0, theme::bold_family())).color(p.on_surface));
            ui.add_space(6.0);
            theme::progress(ui, &self.theme, s.progress() as f32, 10.0);
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(self.theme.label(format!("{:.1}%", s.progress() * 100.0)));
                ui.label(self.theme.label(crate::tf!("эпоха {}", "epoch {}",
                    s.epoch.map(|e| e.to_string()).unwrap_or_else(|| "—".into())
                )));
                ui.label(self.theme.label(crate::tf!("сохранений {}", "saves {}", s.saves)));
            });
        });

        ui.add_space(12.0);
        let loss = s.loss.map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into());
        let lr = s.lr.map(|v| format!("{v:.2e}")).unwrap_or_else(|| "—".into());
        let ram = s.ram_mb.map(|v| format!("{v}")).unwrap_or_else(|| "—".into());
        let threads = s.threads.map(|v| v.to_string()).unwrap_or_else(|| "—".into());
        let profile = s.profile.clone().unwrap_or_else(|| self.form.preset.clone());
        let cards = [
            ("Loss", loss),
            ("LR", lr),
            (tr("RAM, МБ"), ram),
            (tr("Потоки"), threads),
            (tr("Профиль"), profile),
        ];
        ui.columns(5, |cols| {
            for (i, (label, value)) in cards.iter().enumerate() {
                cols[i].set_width(cols[i].available_width());
                self.theme
                    .card(p.surface_container_high, 16.0)
                    .inner_margin(egui::Margin::symmetric(14.0, 10.0))
                    .show(&mut cols[i], |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(self.theme.label(*label));
                        ui.label(self.theme.value(value.clone()));
                    });
            }
        });

        ui.add_space(12.0);
        self.ui_log(ui, 260.0);
    }

    fn ui_done(&mut self, ui: &mut egui::Ui) {
        let p = self.theme.pal;
        ui.set_max_width(860.0);
        let code = self.exit_code.unwrap_or(-1);
        self.theme.card(p.surface_container_low, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(
                RichText::new(if code == 0 { tr("Готово") } else { tr("Остановлено") })
                    .font(egui::FontId::new(28.0, theme::bold_family()))
                    .color(if code == 0 { p.primary } else { p.error }),
            );
            ui.add_space(4.0);
            let loss = self.stats.loss.map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into());
            ui.label(self.theme.label(crate::tf!("шагов {}, loss {}, сохранений {}", "steps {}, loss {}, saves {}",
                self.stats.step, loss, self.stats.saves
            )));
            ui.label(self.theme.label(crate::tf!("адаптеры: {}/final_model", "adapters: {}/final_model", self.form.out)));
        });
        ui.add_space(12.0);
        self.ui_log(ui, 280.0);
    }

    fn ui_log(&mut self, ui: &mut egui::Ui, height: f32) {
        let p = self.theme.pal;
        self.theme.card(p.surface_container_lowest, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.label(self.theme.label(tr("Журнал")));
            ScrollArea::vertical()
                .max_height(height)
                .stick_to_bottom(true)
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    let skip = self.log.len().saturating_sub(200);
                    for line in self.log.iter().skip(skip) {
                        ui.label(RichText::new(line).monospace().size(12.0).color(p.on_surface_variant));
                    }
                });
        });
    }
}

/// M3 filled text field: тональный контейнер + скругление 16.
fn text_field(ui: &mut egui::Ui, theme: &Theme, value: &mut String, hint: &str, width: Option<f32>) {
    let p = theme.pal;
    let mut edit = egui::TextEdit::singleline(value)
        .frame(false)
        .text_color(p.on_surface)
        .margin(egui::Margin::symmetric(0.0, 7.0))
        .hint_text(RichText::new(hint).color(p.on_surface_variant));
    edit = match width {
        None => edit.desired_width(f32::INFINITY),
        Some(w) => edit.desired_width(w),
    };
    let resp = ui.add(edit);
    let color = if resp.has_focus() { p.primary } else { p.outline_variant };
    ui.painter().line_segment(
        [
            egui::pos2(resp.rect.left(), resp.rect.bottom() + 3.0),
            egui::pos2(resp.rect.right(), resp.rect.bottom() + 3.0),
        ],
        egui::Stroke::new(1.0_f32, color),
    );
}

fn text_row(ui: &mut egui::Ui, theme: &Theme, label: &str, value: &mut String, hint: &str) {
    ui.label(theme.label(label));
    text_field(ui, theme, value, hint, None);
    ui.add_space(6.0);
}

fn pair(
    ui: &mut egui::Ui,
    theme: &Theme,
    l1: &str,
    v1: &mut String,
    h1: &str,
    l2: &str,
    v2: &mut String,
    h2: &str,
) {
    ui.horizontal(|ui| {
        let width = (ui.available_width() - 16.0) / 2.0;
        ui.vertical(|ui| {
            ui.set_width(width);
            text_row(ui, theme, l1, v1, h1);
        });
        ui.vertical(|ui| {
            ui.set_width(width);
            text_row(ui, theme, l2, v2, h2);
        });
    });
}

fn combo_row(ui: &mut egui::Ui, theme: &Theme, label: &str, value: &mut String, options: &[&str]) {
    ui.label(theme.label(label));
    let p = theme.pal;
    let resp = ui
        .scope(|ui| {
            let v = &mut ui.visuals_mut().widgets;
            v.inactive.weak_bg_fill = egui::Color32::TRANSPARENT;
            v.inactive.bg_fill = egui::Color32::TRANSPARENT;
            v.hovered.weak_bg_fill = p.surface_container_high;
            egui::ComboBox::from_id_salt(label)
                .selected_text(value.clone())
                .width(ui.available_width())
                .show_ui(ui, |ui| {
                    for opt in options {
                        ui.selectable_value(value, (*opt).to_string(), *opt);
                    }
                })
                .response
        })
        .inner;
    ui.painter().line_segment(
        [
            egui::pos2(resp.rect.left(), resp.rect.bottom() + 3.0),
            egui::pos2(resp.rect.right(), resp.rect.bottom() + 3.0),
        ],
        egui::Stroke::new(1.0_f32, p.outline_variant),
    );
    ui.add_space(6.0);
}

pub fn run(base: Launch, autostart: bool, start_tab: &str, theme_name: Option<&str>) -> Result<(), String> {
    let st = crate::settings::load();
    crate::i18n::set_lang(&st.lang);
    // Material You: seed по умолчанию — системный акцент.
    let seed = crate::settings::resolve_seed(&st.seed);
    let (dark, explicit) = match theme_name {
        Some("dark") => (true, true),
        Some("light") => (false, true),
        _ => (st.dark, false),
    };
    let mut st = st;
    if explicit {
        st.dark = dark;
        crate::settings::save(&st);
    }
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1180.0, 760.0])
            .with_min_inner_size([900.0, 620.0])
            // Своя M3-шапка вместо нативной рамки Windows.
            .with_decorations(false)
            .with_title("Local Forge"),
        ..Default::default()
    };
    eframe::run_native(
        "Local Forge",
        options,
        Box::new(move |cc| {
            let mut app = GuiApp::new(cc, base);
            app.seed = seed;
            app.settings = st;
            app.theme_dark = dark;
            app.set_theme(dark);
            app.apply_settings();
            app.theme.apply(&cc.egui_ctx);
            app.tab = match start_tab {
                "train" => Tab::Train,
                "download" => Tab::Download,
                "settings" => Tab::Settings,
                _ => Tab::Chat,
            };
            if start_tab == "agent" {
                app.chat.agent = true;
            }
            if app.tab == Tab::Train {
                app.start_check();
            }
            if autostart {
                app.tab = Tab::Train;
                app.start_clicked();
            }
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| crate::tf!("не удалось открыть окно: {e}", "failed to open the window: {e}"))
}

#[cfg(test)]
mod command_tests {
    use super::command_flags;

    #[test]
    fn parses_training_flags() {
        let flags = command_flags(
            "python main.py --model models/gemma_base --epochs 2 --packing \
             --lr=0.0003 --data \"database/train 1.jsonl\" --no_amp --kd \
             --kd_weight 0.7 --kd_top_k 12 --unknown value",
        );
        let get = |k: &str| {
            flags
                .iter()
                .find(|(key, _)| key == k)
                .cloned()
                .unwrap_or_else(|| panic!("{}", crate::tf!("нет флага {k}", "no flag {k}")))
        };
        assert_eq!(get("model").1.as_deref(), Some("models/gemma_base"));
        assert_eq!(get("epochs").1.as_deref(), Some("2"));
        assert_eq!(get("lr").1.as_deref(), Some("0.0003"));
        assert_eq!(get("data").1.as_deref(), Some("database/train 1.jsonl"));
        assert_eq!(get("packing").1, None);
        assert_eq!(get("no_amp").1, None);
        assert_eq!(get("kd").1, None);
        assert_eq!(get("kd_weight").1.as_deref(), Some("0.7"));
        assert_eq!(get("kd_top_k").1.as_deref(), Some("12"));
        assert_eq!(get("unknown").1.as_deref(), Some("value"));
    }
}
