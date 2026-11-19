//! forge-train — простой запуск дообучения Forge.
//!
//! Приложение не заменяет Python-движок, а делает работу с ним удобной:
//! одна команда, авто-настройки, живой прогресс и управление на лету.
//! Основной интерфейс — окно в стиле Material 3 (Material You), плюс
//! терминальные режимы для SSH/скриптов.

// В release это оконное приложение: двойной клик не должен открывать консоль.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod chat;
mod checkpoints;
mod color;
mod doctor;
mod downloads;
mod folders;
mod gui;
mod i18n;
mod models;
mod parse;
mod runner;
mod settings;
mod sysinfo;
mod theme;
mod ui;
mod validate;

use crate::i18n::tr;
use clap::{Args, Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use runner::Runner;

#[derive(Parser)]
#[command(
    name = "forge-train",
    version,
    about = tr("Запуск дообучения Forge: одно окно и живой прогресс")
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Запустить дообучение
    Run(RunArgs),
    /// Проверить окружение и показать автоплан
    Doctor(DoctorArgs),
    /// Показать чекпоинты
    Checkpoints(CheckpointsArgs),
}

#[derive(Args, Clone)]
struct Common {
    /// Папка с Python-движком (по умолчанию ./python рядом с exe или в cwd)
    #[arg(long)]
    engine: Option<PathBuf>,
    /// Рабочая папка проекта: здесь лежат models/, database/ и output
    #[arg(long)]
    project: Option<PathBuf>,
    /// Интерпретатор Python (по умолчанию — первый python.exe из PATH)
    #[arg(long)]
    python: Option<String>,
}

#[derive(Args)]
struct RunArgs {
    #[command(flatten)]
    common: Common,

    /// Модель (папка с safetensors). По умолчанию — как в config.py
    #[arg(long)]
    model: Option<String>,

    /// Датасет jsonl. По умолчанию database/train.jsonl или train_11.jsonl
    #[arg(long)]
    data: Option<String>,

    /// Валидационный jsonl (на обучение не влияет)
    #[arg(long, default_value = "database/validation.jsonl")]
    val_file: String,

    /// Файл личности для подмешивания
    #[arg(long, default_value = "database/identity.jsonl")]
    identity_file: String,

    /// Куда писать адаптеры и чекпоинты
    #[arg(long, default_value = "output_model")]
    out: String,

    /// Профиль нагрузки
    #[arg(long, default_value = "work", value_parser = ["game", "work", "max", "turbo"])]
    preset: String,

    #[arg(long, default_value_t = 1)]
    epochs: u32,
    #[arg(long, default_value_t = 2e-5)]
    lr: f64,
    #[arg(long, default_value_t = 1.0)]
    max_grad_norm: f64,
    #[arg(long, default_value_t = 0.05)]
    warmup_ratio: f64,
    #[arg(long, default_value_t = 50)]
    save_steps: u32,

    #[arg(long)]
    max_samples: Option<u64>,
    #[arg(long)]
    max_length: Option<u64>,
    #[arg(long)]
    batch_size: Option<u64>,
    #[arg(long)]
    grad_accum: Option<u64>,
    #[arg(long)]
    lora_r: Option<u64>,
    #[arg(long)]
    lora_alpha: Option<u64>,

    #[arg(long, default_value_t = 0.0)]
    lora_plus_ratio: f64,
    #[arg(long, default_value = "lora", value_parser = ["lora", "full"])]
    tune_mode: String,

    #[arg(long, default_value = "auto", value_parser = ["auto", "bf16", "fp16", "fp32"])]
    dtype: String,
    #[arg(long = "no-amp", action = clap::ArgAction::SetTrue)]
    no_amp: bool,
    #[arg(long, default_value = "auto", value_parser = ["auto", "bf16", "fp16"])]
    amp_dtype: String,
    #[arg(long = "no-grad-checkpointing", action = clap::ArgAction::SetTrue)]
    no_grad_checkpointing: bool,

    #[arg(long)]
    optimizer: Option<String>,
    #[arg(long)]
    lisa_layers: Option<u64>,
    #[arg(long)]
    lisa_interval: Option<u64>,
    #[arg(long, default_value = "auto", value_parser = ["auto", "on", "off"])]
    quant4: String,
    #[arg(long, default_value = "auto", value_parser = ["auto", "sdpa", "fa2"])]
    attn: String,
    #[arg(long)]
    liger: bool,
    #[arg(long)]
    compile: bool,

    #[arg(long, default_value = "")]
    replay_file: String,
    #[arg(long, default_value_t = 0.2)]
    replay_ratio: f64,
    #[arg(long, default_value_t = 0.0)]
    kl_anchor: f64,

    #[arg(long, default_value = "sft", value_parser = ["sft", "orpo", "heretic"])]
    objective: String,
    #[arg(long, default_value = "")]
    orpo_file: String,
    #[arg(long, default_value_t = 0.1)]
    orpo_beta: f64,

    /// Heretic: число trials ARA
    #[arg(long, default_value_t = 3)]
    heretic_trials: u32,
    /// Heretic: число случайных startup trials
    #[arg(long, default_value_t = 1)]
    heretic_startup_trials: u32,
    /// Heretic: auto (CPU->bnb4, CUDA->none), none, bnb4
    #[arg(long, default_value = "auto", value_parser = ["auto", "none", "bnb4"])]
    heretic_quant: String,

    /// Сортировать сэмплы по длине внутри окна (меньше padding, быстрее)
    #[arg(long)]
    group_by_length: bool,

    /// Склеивать диалоги в окно max_length (на CPU обычно невыгодно)
    #[arg(long)]
    packing: bool,

    /// Дистилляция: KL-лосс по top-k шансам учителя (поле soft в данных)
    #[arg(long)]
    kd: bool,
    #[arg(long, default_value_t = 0.5)]
    kd_weight: f64,
    #[arg(long, default_value_t = 8)]
    kd_top_k: u32,

    /// Не кэшировать токенизацию
    #[arg(long = "no-cache", action = clap::ArgAction::SetTrue)]
    no_cache: bool,

    /// Начать без загрузки сохранённого состояния
    #[arg(long)]
    no_resume_state: bool,

    /// Терминальный режим без окна (транслировать вывод)
    #[arg(long)]
    plain: bool,

    /// Старый терминальный дашборд вместо окна
    #[arg(long)]
    tui: bool,

    /// Сразу начать обучение, без экрана настроек
    #[arg(long)]
    start: bool,

    /// С какой вкладки открыть окно
    #[arg(long, default_value = "chat", value_parser = ["chat", "agent", "train", "download", "settings"])]
    tab: String,

    /// Тема оформления (по умолчанию — сохранённая в настройках)
    #[arg(long, value_parser = ["light", "dark"])]
    theme: Option<String>,

    /// Дополнительные аргументы main.py — передаются как есть (после --)
    #[arg(last = true)]
    extra: Vec<String>,
}

#[derive(Args)]
struct DoctorArgs {
    #[command(flatten)]
    common: Common,
    #[arg(long)]
    model: Option<String>,
}

#[derive(Args)]
struct CheckpointsArgs {
    #[arg(long, default_value = "output_model")]
    out: PathBuf,
}

/// Всё, что нужно, чтобы запустить обучение: пути + аргументы.
#[derive(Clone)]
pub struct Launch {
    pub python: String,
    pub engine: PathBuf,
    pub ws: PathBuf,
    pub model: String,
    pub data: String,
    pub val_file: String,
    pub identity_file: String,
    pub out: String,
    pub preset: String,
    pub epochs: u32,
    pub lr: f64,
    pub max_grad_norm: f64,
    pub warmup_ratio: f64,
    pub save_steps: Option<u32>,
    pub max_samples: Option<u64>,
    pub max_length: Option<u32>,
    pub batch_size: Option<u32>,
    pub grad_accum: Option<u32>,
    pub lora_r: Option<u32>,
    pub lora_alpha: Option<u32>,
    pub lora_plus_ratio: f64,
    pub tune_mode: String,
    pub dtype: String,
    pub amp: bool,
    pub amp_dtype: Option<String>,
    pub grad_checkpointing: bool,
    pub optimizer: Option<String>,
    pub lisa_layers: Option<u32>,
    pub lisa_interval: Option<u32>,
    pub quant4: String,
    pub attn: String,
    pub liger: bool,
    pub compile: bool,
    pub replay_file: String,
    pub replay_ratio: f64,
    pub kl_anchor: f64,
    pub objective: String,
    pub orpo_file: String,
    pub orpo_beta: f64,
    pub heretic_trials: u32,
    pub heretic_startup_trials: u32,
    pub heretic_quant: String,
    pub group_by_length: bool,
    pub packing: bool,
    pub kd: bool,
    pub kd_weight: f64,
    pub kd_top_k: u32,
    pub cache: bool,
    pub no_resume_state: bool,
    pub extra: Vec<String>,
}

fn push_num(args: &mut Vec<String>, flag: &str, value: Option<u64>) {
    if let Some(v) = value {
        args.push(flag.to_string());
        args.push(v.to_string());
    }
}

impl Launch {
    pub fn args(&self) -> Vec<String> {
        let main_py = self.engine.join("main.py");
        let mut args: Vec<String> = vec![main_py.to_string_lossy().into_owned()];

        args.push("--model".into());
        args.push(self.model.clone());
        if !self.data.is_empty() {
            args.push("--train_file".into());
            args.push(self.data.clone());
        }
        if !self.val_file.is_empty() {
            args.push("--val_file".into());
            args.push(self.val_file.clone());
        }
        if !self.identity_file.is_empty() {
            args.push("--identity_file".into());
            args.push(self.identity_file.clone());
        }
        args.push("--output_dir".into());
        args.push(self.out.clone());
        args.push("--preset".into());
        args.push(self.preset.clone());

        push_num(&mut args, "--epochs", Some(self.epochs as u64));
        args.push("--lr".into());
        args.push(format!("{}", self.lr));
        args.push("--max_grad_norm".into());
        args.push(format!("{}", self.max_grad_norm));
        args.push("--warmup_ratio".into());
        args.push(format!("{}", self.warmup_ratio));

        push_num(&mut args, "--save_steps", self.save_steps.map(|v| v as u64));
        push_num(&mut args, "--max_samples", self.max_samples);
        push_num(&mut args, "--max_length", self.max_length.map(|v| v as u64));
        push_num(&mut args, "--batch_size", self.batch_size.map(|v| v as u64));
        push_num(&mut args, "--grad_accum", self.grad_accum.map(|v| v as u64));
        push_num(&mut args, "--lora_r", self.lora_r.map(|v| v as u64));
        push_num(&mut args, "--lora_alpha", self.lora_alpha.map(|v| v as u64));

        if self.lora_plus_ratio > 0.0 {
            args.push("--lora_plus_ratio".into());
            args.push(format!("{}", self.lora_plus_ratio));
        }
        args.push("--tune_mode".into());
        args.push(self.tune_mode.clone());

        args.push("--dtype".into());
        args.push(self.dtype.clone());
        if !self.amp {
            args.push("--no_amp".into());
        }
        if let Some(dt) = &self.amp_dtype {
            args.push("--amp_dtype".into());
            args.push(dt.clone());
        }
        if !self.grad_checkpointing {
            args.push("--no_grad_checkpointing".into());
        }

        if let Some(opt) = &self.optimizer {
            args.push("--optimizer".into());
            args.push(opt.clone());
        }
        push_num(&mut args, "--lisa_layers", self.lisa_layers.map(|v| v as u64));
        push_num(&mut args, "--lisa_interval", self.lisa_interval.map(|v| v as u64));
        args.push("--quant4".into());
        args.push(self.quant4.clone());
        args.push("--attn".into());
        args.push(self.attn.clone());
        if self.liger {
            args.push("--liger".into());
        }
        if self.compile {
            args.push("--compile".into());
        }

        if !self.replay_file.is_empty() {
            args.push("--replay_file".into());
            args.push(self.replay_file.clone());
            args.push("--replay_ratio".into());
            args.push(format!("{}", self.replay_ratio));
        }
        if self.kl_anchor > 0.0 {
            args.push("--kl_anchor".into());
            args.push(format!("{}", self.kl_anchor));
        }

        if self.objective == "orpo" {
            args.push("--objective".into());
            args.push("orpo".into());
            if !self.orpo_file.is_empty() {
                args.push("--orpo_file".into());
                args.push(self.orpo_file.clone());
            }
            args.push("--orpo_beta".into());
            args.push(format!("{}", self.orpo_beta));
        }
        if self.objective == "heretic" {
            args.push("--objective".into());
            args.push("heretic".into());
            args.push("--heretic_trials".into());
            args.push(self.heretic_trials.to_string());
            args.push("--heretic_startup_trials".into());
            args.push(self.heretic_startup_trials.to_string());
            args.push("--heretic_quant".into());
            args.push(self.heretic_quant.clone());
        }

        if self.group_by_length {
            args.push("--group_by_length".into());
        }
        if self.packing {
            args.push("--packing".into());
        }
        if self.kd {
            args.push("--kd".into());
            args.push("--kd_weight".into());
            args.push(format!("{}", self.kd_weight));
            args.push("--kd_top_k".into());
            args.push(self.kd_top_k.to_string());
        }
        if !self.cache {
            args.push("--no_cache".into());
        }
        if self.no_resume_state {
            args.push("--no_resume_state".into());
        }
        args.extend(self.extra.iter().cloned());
        args
    }

    pub fn cmdline(&self) -> String {
        format!(
            "{} {}",
            self.python,
            self.args()
                .iter()
                .map(|x| quote_if_needed(x))
                .collect::<Vec<_>>()
                .join(" ")
        )
    }

    pub fn spawn(&self) -> std::io::Result<Runner> {
        Runner::spawn(&self.python, &self.args(), &self.ws)
    }
}

fn main() -> ExitCode {
    install_panic_hook();

    let mut argv: Vec<String> = std::env::args().collect();
    // Двойной клик (или флаги без подкоманды) — это запуск окна.
    let known = ["run", "doctor", "checkpoints", "help"];
    let has_sub = argv.get(1).map(|a| known.contains(&a.as_str())).unwrap_or(false);
    let is_help = argv
        .get(1)
        .map(|a| matches!(a.as_str(), "-h" | "--help" | "-V" | "--version"))
        .unwrap_or(false);
    if !has_sub && !is_help {
        argv.insert(1, "run".into());
    }
    let cli = match Cli::try_parse_from(argv) {
        Ok(cli) => cli,
        Err(e) => {
            let _ = e.print();
            report_error(&e.to_string());
            return ExitCode::from(2);
        }
    };
    let result = match cli.cmd {
        Cmd::Run(a) => run(a),
        Cmd::Doctor(a) => run_doctor(a),
        Cmd::Checkpoints(a) => {
            checkpoints::print(&a.out);
            Ok(0)
        }
    };
    match result {
        Ok(code) => ExitCode::from(code.clamp(0, 255) as u8),
        Err(msg) => {
            report_error(&msg);
            ExitCode::from(2)
        }
    }
}

fn is_project(p: &Path) -> bool {
    p.join("models").is_dir() || p.join("database").is_dir()
}

fn workspace(given: &Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(p) = given {
        return Ok(p.clone());
    }
    let cwd = std::env::current_dir().map_err(|e| crate::tf!("не смог определить рабочую папку: {e}", "failed to determine the working folder: {e}"))?;
    if is_project(&cwd) {
        return Ok(cwd);
    }
    // Двойной клик: cwd — папка с exe (например "Learning on CPU" внутри проекта).
    // Тогда рабочей папкой берём корень проекта рядом с ней.
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            for cand in [dir.to_path_buf(), dir.join("..")] {
                if is_project(&cand) {
                    return Ok(cand);
                }
            }
        }
    }
    Ok(cwd)
}

/// Есть ли у процесса консоль (при запуске двойным кликом её нет).
#[cfg(windows)]
fn has_console() -> bool {
    extern "system" {
        fn GetConsoleWindow() -> *mut std::ffi::c_void;
    }
    unsafe { !GetConsoleWindow().is_null() }
}

#[cfg(not(windows))]
fn has_console() -> bool {
    true
}

fn error_log_path() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    Some(exe.parent()?.join("forge-train-error.log"))
}

fn log_error(msg: &str) {
    if let Some(path) = error_log_path() {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "{msg}");
        }
    }
}

/// Ошибку нужно показать, даже если консоли нет: иначе окно просто не откроется.
fn report_error(msg: &str) {
    log_error(msg);
    if has_console() {
        crate::tf!("Ошибка: {msg}", "Error: {msg}");
    } else {
        show_message_box("Forge Train", msg);
    }
}

#[cfg(windows)]
fn show_message_box(title: &str, text: &str) {
    use std::ffi::c_void;
    extern "system" {
        fn MessageBoxW(hwnd: *mut c_void, text: *const u16, caption: *const u16, utype: u32) -> i32;
    }
    let text: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
    let title: Vec<u16> = title.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), 0x10);
    }
}

#[cfg(not(windows))]
fn show_message_box(_title: &str, text: &str) {
    eprintln!("{text}");
}

fn install_panic_hook() {
    std::panic::set_hook(Box::new(|info| {
        let msg = crate::tf!("Паника: {info}", "Panic: {info}");
        log_error(&msg);
        if has_console() {
            eprintln!("{msg}");
        } else {
            show_message_box(tr("Forge Train — ошибка"), &msg);
        }
    }));
}

fn resolve_engine(given: &Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(p) = given {
        return Ok(p.clone());
    }
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            // рядом с exe (обычная установка)
            candidates.push(dir.join("python"));
            // ../..: exe в target/debug или target/release при cargo run
            candidates.push(dir.join("..").join("..").join("python"));
        }
    }
    candidates.push(PathBuf::from("python"));
    for cand in candidates {
        if cand.join("main.py").exists() {
            return Ok(cand);
        }
    }
    Err(tr("не нашёл движок (main.py). Укажи --engine <папка с main.py>").into())
}

fn resolve_data(given: &Option<String>, ws: &Path) -> String {
    if let Some(d) = given {
        return d.clone();
    }
    for cand in ["database/train.jsonl", "database/train_11.jsonl"] {
        if ws.join(cand).exists() {
            return cand.to_string();
        }
    }
    "database/train.jsonl".to_string()
}

fn default_model(ws: &Path) -> String {
    for cand in ["models/gemma_base", "models/qwen2.5_3b"] {
        if ws.join(cand).join("model.safetensors").exists() {
            return cand.to_string();
        }
    }
    "models/gemma_base".to_string()
}

/// Путь к настоящему python.exe.
///
/// Строку "python" отдавать нельзя: когда рабочая папка — корень проекта,
/// Windows первым находит лежащий там пустой файл `python` без расширения
/// и падает с «не удалось запустить».
fn resolve_python(explicit: Option<&String>) -> String {
    if let Some(p) = explicit {
        let p = p.trim();
        if !p.is_empty() {
            return p.to_string();
        }
    }
    // python.txt рядом с exe — ручное указание интерпретатора
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Ok(text) = std::fs::read_to_string(dir.join("python.txt")) {
                if let Some(line) = text.lines().next() {
                    let line = line.trim().trim_matches('"');
                    if !line.is_empty() && Path::new(line).is_file() {
                        return line.to_string();
                    }
                }
            }
            // Вендинсталляция из setup.bat: <корень>/.venv
            for cand in [dir.join(".venv"), dir.join("python").join(".venv")] {
                let py = cand.join("Scripts").join("python.exe");
                if py.is_file() {
                    return py.to_string_lossy().into_owned();
                }
            }
        }
    }
    // PATH: только реальные .exe, в порядке PATH
    if let Some(paths) = std::env::var_os("PATH") {
        let mut py_launcher: Option<PathBuf> = None;
        for dir in std::env::split_paths(&paths) {
            for name in ["python.exe", "python3.exe"] {
                let cand = dir.join(name);
                if cand.is_file() {
                    return cand.to_string_lossy().into_owned();
                }
            }
            if py_launcher.is_none() {
                let py = dir.join("py.exe");
                if py.is_file() {
                    py_launcher = Some(py);
                }
            }
        }
        if let Some(py) = py_launcher {
            return py.to_string_lossy().into_owned();
        }
    }
    "python".to_string()
}

fn quote_if_needed(s: &str) -> String {
    if s.contains(' ') {
        format!("\"{s}\"")
    } else {
        s.to_string()
    }
}

fn build_launch(a: &RunArgs) -> Result<Launch, String> {
    let ws = std::path::absolute(workspace(&a.common.project)?)
        .map_err(|e| crate::tf!("не смог определить рабочую папку: {e}", "failed to determine the working folder: {e}"))?;
    let engine = resolve_engine(&a.common.engine)?;
    if !engine.join("main.py").exists() {
        return Err(crate::tf!("в '{}' нет main.py", "'{}' has no main.py", engine.display()));
    }
    let model = a.model.clone().unwrap_or_else(|| default_model(&ws));
    let data = resolve_data(&a.data, &ws);
    Ok(Launch {
        python: resolve_python(a.common.python.as_ref()),
        engine,
        ws,
        model,
        data,
        val_file: a.val_file.clone(),
        identity_file: a.identity_file.clone(),
        out: a.out.clone(),
        preset: a.preset.clone(),
        epochs: a.epochs,
        lr: a.lr,
        max_grad_norm: a.max_grad_norm,
        warmup_ratio: a.warmup_ratio,
        save_steps: Some(a.save_steps),
        max_samples: a.max_samples,
        max_length: a.max_length.map(|v| v as u32),
        batch_size: a.batch_size.map(|v| v as u32),
        grad_accum: a.grad_accum.map(|v| v as u32),
        lora_r: a.lora_r.map(|v| v as u32),
        lora_alpha: a.lora_alpha.map(|v| v as u32),
        lora_plus_ratio: a.lora_plus_ratio,
        tune_mode: a.tune_mode.clone(),
        dtype: a.dtype.clone(),
        amp: !a.no_amp,
        amp_dtype: if a.amp_dtype == "auto" { None } else { Some(a.amp_dtype.clone()) },
        grad_checkpointing: !a.no_grad_checkpointing,
        optimizer: a.optimizer.clone(),
        lisa_layers: a.lisa_layers.map(|v| v as u32),
        lisa_interval: a.lisa_interval.map(|v| v as u32),
        quant4: a.quant4.clone(),
        attn: a.attn.clone(),
        liger: a.liger,
        compile: a.compile,
        replay_file: a.replay_file.clone(),
        replay_ratio: a.replay_ratio,
        kl_anchor: a.kl_anchor,
        objective: a.objective.clone(),
        orpo_file: a.orpo_file.clone(),
        orpo_beta: a.orpo_beta,
        heretic_trials: a.heretic_trials,
        heretic_startup_trials: a.heretic_startup_trials,
        heretic_quant: a.heretic_quant.clone(),
        group_by_length: a.group_by_length,
        packing: a.packing,
        kd: a.kd,
        kd_weight: a.kd_weight,
        kd_top_k: a.kd_top_k,
        cache: !a.no_cache,
        no_resume_state: a.no_resume_state,
        extra: a.extra.clone(),
    })
}

fn run(a: RunArgs) -> Result<i32, String> {
    let mut launch = build_launch(&a)?;
    // Пресет из настроек — если не задан явно в командной строке.
    if !std::env::args().any(|x| x.starts_with("--preset")) {
        launch.preset = settings::load().train_preset;
    }

    // Окно — основной режим. --plain и --tui оставлены для SSH/скриптов.
    if !a.plain && !a.tui {
        match gui::run(launch.clone(), a.start, &a.tab, a.theme.as_deref()) {
            Ok(()) => return Ok(0),
            Err(e) => {
                crate::tf!("Не удалось открыть окно ({e}) — запускаю в терминале.", "Failed to open the window ({e}) — starting in the terminal.");
            }
        }
    }

    let runner = launch
        .spawn()
        .map_err(|e| crate::tf!("не удалось запустить '{}': {e}", "failed to start '{}': {e}", launch.python))?;

    let cmdline = launch.cmdline();
    let mut app = ui::App::new();
    app.stats.model = Some(launch.model.clone());
    app.push_log(format!("$ {cmdline}"));
    app.push_log(crate::tf!("рабочая папка: {}", "working folder: {}", launch.ws.display()));
    app.push_log(crate::tf!("датасет: {}", "dataset: {}", launch.data));

    let use_tui = a.tui && ui::stdout_is_tty();
    let code = if use_tui {
        ui::run_tui(runner, app).map_err(|e| crate::tf!("дашборд упал: {e}", "dashboard crashed: {e}"))?
    } else {
        if !ui::stdout_is_tty() {
            println!("$ {cmdline}");
            crate::tf!("(вывод перенаправлен — обычный режим, дашборд отключён)", "(output redirected — normal mode, dashboard disabled)");
        }
        ui::run_plain(runner, app).map_err(|e| crate::tf!("ошибка запуска: {e}", "launch error: {e}"))?
    };

    println!();
    if code == 0 {
        crate::tf!("Готово. Адаптеры и чекпоинты: {}", "Done. Adapters and checkpoints: {}", launch.out);
        println!("{}", crate::tf!(
            "Проверить в чате:  python python/test_chat.py --base <модель> --lora {}/final_model",
            "Check in chat:  python python/test_chat.py --base <model> --lora {}/final_model",
            launch.out
        ));
    } else {
        crate::tf!("Процесс завершился с кодом {code}. Смотри вывод выше.", "Process finished with code {code}. See the output above.");
    }
    Ok(code)
}

fn run_doctor(a: DoctorArgs) -> Result<i32, String> {
    let ws = std::path::absolute(workspace(&a.common.project)?)
        .map_err(|e| crate::tf!("не смог определить рабочую папку: {e}", "failed to determine the working folder: {e}"))?;
    let engine = resolve_engine(&a.common.engine)?;
    if !engine.join("main.py").exists() {
        return Err(crate::tf!("в '{}' нет main.py — укажи папку с движком", "'{}' has no main.py — specify the engine folder", engine.display()));
    }
    let model = a.model.clone().unwrap_or_else(|| default_model(&ws));
    let model_path = {
        let p = PathBuf::from(&model);
        if p.is_absolute() {
            p
        } else {
            ws.join(p)
        }
    };
    let python = resolve_python(a.common.python.as_ref());
    doctor::run(&python, &engine, &model_path)
        .map_err(|e| crate::tf!("не удалось запустить '{python}': {e}", "failed to start '{python}': {e}"))
}
