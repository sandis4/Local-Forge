//! Проверка настроек под железо: запускает python/forge_plan.py и читает отчёт.

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};
#[cfg(windows)]
use std::os::windows::process::CommandExt;

use serde::{Deserialize, Serialize};

/// Сериализуемое зеркало формы GUI (`Form`). Уходит в forge_plan.py;
/// дефолты здесь — то, что видит план, а не то, что уходит тренеру.
#[derive(Serialize, Clone, PartialEq)]
pub struct Settings {
    pub model: String,
    pub train_file: String,
    pub val_file: String,
    pub out: String,
    pub identity_file: String,
    pub epochs: u32,
    pub lr: f64,
    pub max_grad_norm: f64,
    pub warmup_ratio: f64,
    pub max_length: u32,
    pub max_samples: Option<u64>,
    pub batch_size: u32,
    pub grad_accum: u32,
    pub save_steps: u32,
    pub lora_r: u32,
    pub lora_alpha: u32,
    pub lora_plus_ratio: f64,
    pub tune_mode: String,
    pub preset: String,
    pub dtype: String,
    pub amp: bool,
    pub amp_dtype: String,
    pub grad_checkpointing: bool,
    pub optimizer: String,
    pub lisa_layers: u32,
    pub lisa_interval: u32,
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
    pub group_by_length: bool,
    pub packing: bool,
    pub cache: bool,
    pub kd: bool,
    pub kd_weight: f64,
    pub kd_top_k: u32,
}

#[derive(Deserialize, Default, Clone)]
#[serde(default)]
pub struct Hardware {
    pub cpu: Option<u32>,
    pub ram_total_gb: Option<f64>,
    pub ram_avail_gb: Option<f64>,
    pub cuda: Option<bool>,
    pub gpu: Option<String>,
    pub vram_gb: Option<f64>,
    pub disk_free_gb: Option<f64>,
    pub torch: Option<String>,
    pub python: Option<String>,
}

#[derive(Deserialize, Default, Clone)]
#[serde(default)]
pub struct Plan {
    pub strategy: Option<String>,
    pub weights_gb: Option<f64>,
    pub max_length: Option<u32>,
    pub batch_size: Option<u32>,
    pub grad_accum: Option<u32>,
    pub optimizer: Option<String>,
    pub lisa_layers: Option<u32>,
    pub lisa_interval: Option<u32>,
    pub lora_r: Option<u32>,
    pub lora_alpha: Option<u32>,
    pub save_steps: Option<u32>,
    pub amp: Option<bool>,
    pub amp_dtype: Option<String>,
    pub gradient_checkpointing: Option<bool>,
    pub dtype: Option<String>,
    pub quant4: Option<String>,
}

#[derive(Deserialize, Default, Clone)]
#[serde(default)]
pub struct Check {
    pub level: String,
    pub field: String,
    pub message: String,
}

#[derive(Deserialize, Default, Clone)]
#[serde(default)]
pub struct Report {
    pub hardware: Hardware,
    pub plan: Plan,
    pub checks: Vec<Check>,
    pub errors: u32,
    pub warnings: u32,
}

impl Report {
    pub fn sorted(&self) -> Vec<&Check> {
        let rank = |lvl: &str| match lvl {
            "error" => 0,
            "warn" => 1,
            "info" => 3,
            _ => 2,
        };
        let mut v: Vec<&Check> = self.checks.iter().collect();
        v.sort_by_key(|c| rank(&c.level));
        v
    }
}

/// Запустить проверку. `ws` — рабочая папка: относительные пути в настройках
/// считаются от неё, как и у тренера.
pub fn run(python: &str, engine: &Path, ws: &Path, settings: &Settings) -> Result<Report, String> {
    // Валидатор живёт в бандле приложения. Если указан движок из корня
    // проекта (--engine без forge_plan.py), берём скрипт рядом с exe.
    let mut script = engine.join("forge_plan.py");
    if !script.exists() {
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let cand = dir.join("python").join("forge_plan.py");
                if cand.exists() {
                    script = cand;
                }
            }
        }
    }
    if !script.exists() {
        return Err(crate::tf!("нет '{}' — нечем проверять настройки", "missing '{}' — nothing to validate settings with", script.display()));
    }
    let payload = serde_json::to_vec(settings).map_err(|e| e.to_string())?;
    let mut cmd = Command::new(python);
    #[cfg(windows)]
    {
        // Иначе при открытии «Обучения» всплывает консоль.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd
        .arg(&script)
        .current_dir(ws)
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| crate::tf!("не удалось запустить {python}: {e}", "failed to start {python}: {e}"))?;
    if let Some(mut si) = child.stdin.take() {
        let _ = si.write_all(&payload);
    }
    let out = child.wait_with_output().map_err(|e| e.to_string())?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let short: String = err.trim().chars().take(300).collect();
        return Err(crate::tf!("проверка упала: {short}", "check crashed: {short}"));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str::<Report>(text.trim()).map_err(|e| {
        let tail: String = text.chars().rev().take(200).collect::<String>().chars().rev().collect();
        crate::tf!("не разобрал отчёт проверки: {e}. Ответ: {tail}", "failed to parse the check report: {e}. Response: {tail}")
    })
}
