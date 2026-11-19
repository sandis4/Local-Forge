//! Список чекпоинтов в папке обучения.

use crate::i18n::tr;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub struct Row {
    pub name: String,
    pub step: Option<u64>,
    pub size_mb: f64,
    pub modified: SystemTime,
}

fn dir_size(path: &Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = fs::read_dir(path) {
        for entry in entries.flatten() {
            let meta = match entry.metadata() {
                Ok(m) => m,
                Err(_) => continue,
            };
            if meta.is_dir() {
                total += dir_size(&entry.path());
            } else {
                total += meta.len();
            }
        }
    }
    total
}

fn step_from_name(name: &str) -> Option<u64> {
    name.strip_prefix("checkpoint-step-")?.parse().ok()
}

pub fn scan(out: &Path) -> Vec<Row> {
    let mut rows = Vec::new();
    if let Ok(entries) = fs::read_dir(out) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            let step = step_from_name(&name);
            let known = step.is_some() || name == "final_model" || name == "interactive_checkpoint";
            if !known {
                continue;
            }
            let modified = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(UNIX_EPOCH);
            rows.push(Row {
                name,
                step,
                size_mb: dir_size(&path) as f64 / (1024.0 * 1024.0),
                modified,
            });
        }
    }

    let state = out.join("training_state").join("training_state.pt");
    if state.exists() {
        let modified = fs::metadata(&state)
            .and_then(|m| m.modified())
            .unwrap_or(UNIX_EPOCH);
        rows.push(Row {
            name: tr("training_state (позиция в данных)").into(),
            step: None,
            size_mb: dir_size(&state) as f64 / (1024.0 * 1024.0),
            modified,
        });
    }

    rows.sort_by(|a, b| match (a.step, b.step) {
        (Some(x), Some(y)) => x.cmp(&y),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.name.cmp(&b.name),
    });
    rows
}

fn ago(t: SystemTime) -> String {
    match SystemTime::now().duration_since(t) {
        Ok(d) => {
            let s = d.as_secs();
            if s < 60 {
                crate::tf!("{s}с назад", "{s}s ago")
            } else if s < 3600 {
                crate::tf!("{}м назад", "{}m ago", s / 60)
            } else if s < 86400 {
                crate::tf!("{}ч назад", "{}h ago", s / 3600)
            } else {
                crate::tf!("{}д назад", "{}d ago", s / 86400)
            }
        }
        Err(_) => "—".into(),
    }
}

pub fn print(out: &Path) {
    if !out.exists() {
        crate::tf!("Папки '{}' нет — обучение ещё не запускалось.", "Folder '{}' does not exist — training has not started yet.", out.display());
        return;
    }
    let rows = scan(out);
    if rows.is_empty() {
        crate::tf!("В '{}' нет чекпоинтов.", "No checkpoints in '{}'.", out.display());
        return;
    }
    println!("{:<40} {:>8}  {:>10}  {}", tr("чекпоинт"), tr("шаг"), tr("размер"), tr("изменён"));
    println!("{}", "-".repeat(80));
    for r in rows {
        let step = r.step.map(|s| s.to_string()).unwrap_or_else(|| "—".into());
        crate::tf!("{:<40} {:>8}  {:>8.1} МБ  {}", "{:<40} {:>8}  {:>8.1} MB  {}",
            r.name,
            step,
            r.size_mb,
            ago(r.modified)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_step_from_dir_name() {
        assert_eq!(step_from_name("checkpoint-step-150"), Some(150));
        assert_eq!(step_from_name("checkpoint-step-x"), None);
        assert_eq!(step_from_name("final_model"), None);
    }
}
