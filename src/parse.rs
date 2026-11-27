//! Разбор строк, которые печатает Python-тренер.

use std::time::Instant;

// Префиксы stdout-протокола тренера. Они всегда русские: GUI парсит их
// независимо от языка интерфейса, поэтому через tr() их гонять нельзя.
const STEP_PREFIX: &str = "[Шаг ";
const SCHEDULER_PREFIX: &str = "[Шедулер]";
const EPOCH_PREFIX: &str = "--- Эпоха ";
const MODEL_PREFIX: &str = "Модель: ";
const SAVE_PREFIX: &str = "[Сохранение]";
const FINISHED_MARK: &str = "Обучение завершено";
const TOTAL_STEPS_MARK: &str = "шагов ~";
const RAM_PREFIX: &str = "RAM процесса: ";
const THREADS_PREFIX: &str = "Потоков: ";
const PROFILE_PREFIX: &str = "Профиль: ";

#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub step: u64,
    pub total_steps: Option<u64>,
    pub epoch: Option<u64>,
    pub epochs: Option<u64>,
    pub loss: Option<f64>,
    pub lr: Option<f64>,
    pub ram_mb: Option<u64>,
    pub threads: Option<u64>,
    pub profile: Option<String>,
    pub padding: Option<String>,
    pub model: Option<String>,
    pub saves: u32,
    pub finished: bool,
    pub error: Option<String>,
    pub started: Option<Instant>,
    pub start_step: u64,
}

impl Stats {
    pub fn begin(&mut self) {
        self.started = Some(Instant::now());
        self.start_step = self.step;
    }

    /// Секунд на шаг по фактическому темпу (None, пока нечего считать).
    pub fn secs_per_step(&self) -> Option<f64> {
        let started = self.started?;
        let done = self.step.saturating_sub(self.start_step);
        if done == 0 {
            return None;
        }
        Some(started.elapsed().as_secs_f64() / done as f64)
    }

    pub fn eta(&self) -> Option<f64> {
        let total = self.total_steps?;
        let per = self.secs_per_step()?;
        let left = total.saturating_sub(self.step);
        Some(left as f64 * per)
    }

    pub fn progress(&self) -> f64 {
        match self.total_steps {
            Some(total) if total > 0 => (self.step as f64 / total as f64).clamp(0.0, 1.0),
            _ => 0.0,
        }
    }

    /// Одна строка вывода тренера. Возвращает true, если это строка шага
    /// (её не обязательно показывать в журнале отдельно — она в статусе).
    pub fn feed(&mut self, line: &str) -> bool {
        let trimmed = line.trim();

        if trimmed.starts_with(STEP_PREFIX) {
            self.parse_step(trimmed);
            return true;
        }
        if trimmed.starts_with(SCHEDULER_PREFIX) {
            if let Some(total) = parse_total_steps(trimmed) {
                self.total_steps = Some(total);
            }
        }
        if let Some(rest) = trimmed.strip_prefix(EPOCH_PREFIX) {
            let rest = rest.trim_end_matches("---").trim();
            let mut parts = rest.split('/');
            self.epoch = parts.next().and_then(|x| x.trim().parse().ok());
            self.epochs = parts.next().and_then(|x| x.trim().parse().ok());
        }
        if let Some(rest) = trimmed.strip_prefix("Padding: ") {
            self.padding = Some(rest.to_string());
        }
        if let Some(rest) = trimmed.strip_prefix("[Padding] ") {
            self.padding = Some(rest.to_string());
        }
        if let Some(rest) = trimmed.strip_prefix(MODEL_PREFIX) {
            self.model = Some(rest.to_string());
        }
        if trimmed.starts_with(SAVE_PREFIX) {
            self.saves += 1;
        }
        if trimmed.contains(FINISHED_MARK) {
            self.finished = true;
        }
        if trimmed.starts_with("Traceback")
            || trimmed.starts_with("RuntimeError")
            || trimmed.contains("CUDA out of memory")
            || trimmed.starts_with("error:")
        {
            self.error = Some(trimmed.to_string());
        }
        false
    }

    fn parse_step(&mut self, line: &str) {
        if let Some(end) = line.find(']') {
            if let Ok(step) = line[STEP_PREFIX.len()..end].trim().parse::<u64>() {
                self.step = step;
            }
        }
        for part in line.split(" | ") {
            let part = part.trim();
            if let Some(v) = field(part, "Loss: ") {
                self.loss = v.parse().ok();
            } else if let Some(v) = field(part, "LR: ") {
                self.lr = v.parse().ok();
            } else if let Some(v) = field(part, RAM_PREFIX) {
                self.ram_mb = v.trim_end_matches("MB").trim().parse().ok();
            } else if let Some(v) = field(part, THREADS_PREFIX) {
                self.threads = v.parse().ok();
            } else if let Some(v) = field(part, PROFILE_PREFIX) {
                self.profile = Some(v.to_string());
            }
        }
    }
}

/// "-> шагов ~37, warmup 1." -> 37
fn parse_total_steps(line: &str) -> Option<u64> {
    let idx = line.find(TOTAL_STEPS_MARK)?;
    let rest = &line[idx + TOTAL_STEPS_MARK.len()..];
    let num: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
    num.parse().ok()
}

/// Значение после ключа внутри части строки.
/// Нужен потому, что первая часть строки шага начинается с "[Шаг N] Loss: ...",
/// а не с самого ключа.
fn field<'a>(part: &'a str, key: &str) -> Option<&'a str> {
    let idx = part.find(key)?;
    Some(part[idx + key.len()..].trim())
}

/// "2ч 14м" / "14м 3с"
pub fn human_duration(secs: f64) -> String {
    let secs = secs.max(0.0) as u64;
    let h = secs / 3600;
    let m = (secs % 3600) / 60;
    let s = secs % 60;
    if h > 0 {
        crate::tf!("{h}ч {m:02}м", "{h}h {m:02}m")
    } else if m > 0 {
        crate::tf!("{m}м {s:02}с", "{m}m {s:02}s")
    } else {
        crate::tf!("{s}с", "{s}s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_real_step_line() {
        let mut st = Stats::default();
        assert!(st.feed("[Шаг 42] Loss: 0.8342 | LR: 1.23e-05 | RAM процесса: 2345MB | Потоков: 8 | Профиль: work"));
        assert_eq!(st.step, 42);
        assert!((st.loss.unwrap() - 0.8342).abs() < 1e-9);
        assert!((st.lr.unwrap() - 1.23e-5).abs() < 1e-12);
        assert_eq!(st.ram_mb, Some(2345));
        assert_eq!(st.threads, Some(8));
        assert_eq!(st.profile.as_deref(), Some("work"));
    }

    #[test]
    fn parses_scheduler_total() {
        let mut st = Stats::default();
        st.feed("[Шедулер] Сэмплов на эпоху: ~27 (файл 24 + 15% identity) -> шагов ~13, warmup 1. LR обнулится на последнем шаге эпохи, а не раньше.");
        assert_eq!(st.total_steps, Some(13));
    }

    #[test]
    fn parses_packing_scheduler_line() {
        let mut st = Stats::default();
        st.feed("[Шедулер] Packing: средний диалог ~82 токенов, в окно 256 влезает ~3.1 диалогов");
        assert_eq!(st.total_steps, None);
        st.feed("[Шедулер] Сэмплов на эпоху: ~27 (файл 24 + 15% identity) -> шагов ~2, warmup 1.");
        assert_eq!(st.total_steps, Some(2));
    }

    #[test]
    fn parses_epoch_padding_model() {
        let mut st = Stats::default();
        st.feed("--- Эпоха 1/3 ---");
        st.feed(" Padding: PACKING — диалоги склеены в окно 256 (блочная маска между документами)");
        st.feed(" Модель: models/gemma_base");
        assert_eq!(st.epoch, Some(1));
        assert_eq!(st.epochs, Some(3));
        assert!(st.padding.unwrap().starts_with("PACKING"));
        assert_eq!(st.model.as_deref(), Some("models/gemma_base"));
    }

    #[test]
    fn detects_finish_and_errors() {
        let mut st = Stats::default();
        st.feed("[Сохранение] Готово! Веса сохранены.");
        st.feed(" Обучение завершено или остановлено.");
        assert_eq!(st.saves, 1);
        assert!(st.finished);

        let mut st2 = Stats::default();
        st2.feed("RuntimeError: [enforce fail] CUDA out of memory");
        assert!(st2.error.is_some());
    }

    #[test]
    fn eta_math() {
        let mut st = Stats::default();
        st.step = 10;
        st.total_steps = Some(20);
        assert_eq!(st.progress(), 0.5);
        st.started = Some(std::time::Instant::now());
        st.start_step = 10;
        assert!(st.eta().is_none() || st.eta().unwrap() >= 0.0);
    }

    #[test]
    fn human_duration_formats() {
        assert_eq!(human_duration(45.0), "45с");
        assert_eq!(human_duration(125.0), "2м 05с");
        assert_eq!(human_duration(3600.0 + 600.0), "1ч 10м");
    }
}
