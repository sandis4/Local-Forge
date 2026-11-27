//! Хаб моделей: поиск на Hugging Face, скачивание файлов/квантов,
//! список локальных моделей и предпросмотр датасетов.

use crate::i18n::tr;
use std::collections::VecDeque;
use std::path::Path;

use eframe::egui::{self, RichText, ScrollArea};
use serde::Deserialize;
use serde_json::Value;

use crate::chat::ModelEntry;
use crate::runner::Runner;
use crate::theme::{self, ButtonKind, Theme};

pub enum HubAction {
    UseModel(String),
    Downloaded,
}

pub struct Item {
    pub id: String,
    pub downloads: u64,
    pub likes: u64,
    pub tags: Vec<String>,
    pub author: String,
    pub updated: String,
    pub created: String,
    pub is_new: bool,
    pub library: String,
    pub quant: String,
    pub file_size: u64,
    pub verified: bool,
}

pub struct Variant {
    pub label: String,
    pub size: u64,
}

pub struct Section {
    pub title: String,
    pub note: String,
    pub start: usize,
}

/// События hf_tools.py: `{"type": "...", ...}`.
#[derive(Deserialize)]
#[serde(tag = "type")]
enum HubEvent {
    #[serde(rename = "profile")]
    Profile {
        #[serde(default)]
        count: usize,
        #[serde(default)]
        author: Option<String>,
    },
    #[serde(rename = "section")]
    Section {
        #[serde(default)]
        title: String,
        #[serde(default)]
        note: String,
    },
    #[serde(rename = "item")]
    Item(ItemPayload),
    #[serde(rename = "files")]
    Files {
        #[serde(default)]
        items: Option<Vec<FilePayload>>,
        #[serde(default)]
        variants: Option<Vec<VariantPayload>>,
    },
    #[serde(rename = "progress")]
    Progress {
        #[serde(default)]
        done: u64,
        #[serde(default)]
        total: u64,
        #[serde(default)]
        file: String,
    },
    #[serde(rename = "readme")]
    Readme {
        #[serde(default)]
        text: Option<String>,
    },
    #[serde(rename = "preview")]
    Preview {
        #[serde(default)]
        rows: Option<Vec<Value>>,
        #[serde(default)]
        file: String,
    },
    #[serde(rename = "done")]
    Done,
    #[serde(rename = "error")]
    Error {
        #[serde(default)]
        message: Option<String>,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Deserialize, Default)]
struct ItemPayload {
    #[serde(default)]
    id: String,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    author: String,
    #[serde(default)]
    updated: String,
    #[serde(default)]
    created: String,
    #[serde(default, rename = "new")]
    is_new: bool,
    #[serde(default)]
    library: String,
    #[serde(default)]
    quant: String,
    #[serde(default)]
    file_size: u64,
    #[serde(default)]
    verified: bool,
}

#[derive(Deserialize, Default)]
struct FilePayload {
    #[serde(default)]
    name: String,
    #[serde(default)]
    size: u64,
}

#[derive(Deserialize, Default)]
struct VariantPayload {
    #[serde(default)]
    label: String,
    #[serde(default)]
    size: u64,
}

#[derive(PartialEq, Clone, Copy)]
enum Action {
    Search,
    Feed,
    Profile,
    Files,
    Download,
    Preview,
    Readme,
}

pub struct Downloads {
    pub kind: String,
    pub query: String,
    pub on_device: bool,
    pub results: Vec<Item>,
    pub selected: Option<usize>,
    pub files: Vec<(String, u64)>,
    pub variants: Vec<Variant>,
    pub file_sel: Option<usize>,
    picker_open: bool,
    picker_pos: Option<egui::Pos2>,
    picker_width: f32,
    pub preview: Vec<String>,
    pub log: VecDeque<String>,
    pub progress: Option<(u64, u64, String)>,
    pub status: String,
    pub busy: bool,
    pub sort: String,
    pub readme: Vec<String>,
    pub sections: Vec<Section>,
    feed_loaded: bool,
    feed_user: String,
    feed_quants: String,
    ram_total: f64,
    cpu: u32,
    profile_open: bool,
    profile_author: Option<String>,
    profile_sort: String,
    profile_count: usize,
    pending_profile: Option<String>,
    feed_backup: Option<(Vec<Item>, Vec<Section>)>,
    pub ram_gb: f32,
    auto_searched: bool,
    auto_select: bool,
    queue: Vec<(Vec<String>, Action)>,
    proc: Option<(Runner, Action)>,
}

impl Downloads {
    pub fn new() -> Self {
        Self {
            kind: "model".into(),
            query: String::new(),
            on_device: false,
            results: Vec::new(),
            selected: None,
            files: Vec::new(),
            variants: Vec::new(),
            file_sel: None,
            picker_open: false,
            picker_pos: None,
            picker_width: 320.0,
            preview: Vec::new(),
            log: VecDeque::new(),
            progress: None,
            status: String::new(),
            busy: false,
            sort: "downloads".into(),
            readme: Vec::new(),
            sections: Vec::new(),
            feed_loaded: false,
            feed_user: String::new(),
            feed_quants: String::new(),
            ram_total: 0.0,
            cpu: 8,
            profile_open: false,
            profile_author: None,
            profile_sort: "downloads".into(),
            profile_count: 0,
            pending_profile: None,
            feed_backup: None,
            ram_gb: 6.5,
            auto_searched: false,
            auto_select: false,
            queue: Vec::new(),
            proc: None,
        }
    }

    fn start(&mut self, python: &str, engine_dir: &Path, ws: &Path, args: &[String], action: Action) {
        if self.proc.is_some() {
            self.queue.push((args.to_vec(), action));
            return;
        }
        let script = engine_dir.join("hf_tools.py");
        let mut full = vec![script.to_string_lossy().into_owned()];
        full.extend(args.iter().cloned());
        match Runner::spawn(python, &full, ws) {
            Ok(runner) => {
                self.proc = Some((runner, action));
                self.busy = true;
                self.status = match action {
                    Action::Search => tr("Ищу на Hugging Face…").into(),
                    Action::Feed => tr("Собираю ленту…").into(),
                    Action::Profile => tr("Загружаю профиль…").into(),
                    Action::Files => tr("Читаю список файлов…").into(),
                    Action::Download => tr("Скачиваю…").into(),
                    Action::Preview => tr("Читаю первые строки…").into(),
                    Action::Readme => tr("Читаю README…").into(),
                };
            }
            Err(e) => self.status = crate::tf!("Не удалось запустить: {e}", "Failed to start: {e}"),
        }
    }

    /// Сброс результатов/выбора/предпросмотра перед новой выборкой.
    /// Секции чистит вызывающий: у поиска это зависит от пустого запроса.
    fn reset_view(&mut self) {
        self.results.clear();
        self.selected = None;
        self.files.clear();
        self.variants.clear();
        self.file_sel = None;
        self.preview.clear();
        self.readme.clear();
    }

    pub fn search(&mut self, python: &str, engine_dir: &Path, ws: &Path) {
        if self.profile_open {
            self.profile_open = false;
            self.profile_author = None;
            self.feed_backup = None;
        }
        self.reset_view();
        let kind = self.kind.clone();
        let query = self.query.trim().to_string();
        let sort = self.sort.clone();
        self.auto_select = true;
        if kind == "model" && query.is_empty() {
            self.feed_loaded = false;
            return;
        }
        self.sections.clear();
        self.feed_loaded = true;
        self.start(
            python,
            engine_dir,
            ws,
            &["search".into(), kind, query, sort],
            Action::Search,
        );
    }

    fn budget_bytes(&self) -> f64 {
        let ram = (self.ram_gb.max(1.0) as f64) * 0.9;
        let cpu = (self.cpu.max(1) as f64) * 1.25;
        ram.min(cpu) * 1e9
    }

    fn default_variant(&self) -> usize {
        if self.variants.is_empty() {
            return 0;
        }
        let budget = self.budget_bytes();
        let mut best = 0;
        for (i, v) in self.variants.iter().enumerate() {
            if (v.size as f64) <= budget {
                best = i;
            }
        }
        best
    }

    pub fn load_feed(&mut self, python: &str, engine_dir: &Path, ws: &Path,
                     user: &str, quants: &str, ram_gb: f32, cpu: u32) {
        self.reset_view();
        self.sections.clear();
        self.feed_loaded = true;
        self.auto_select = true;
        let args = vec![
            "feed".to_string(),
            "--user".to_string(),
            user.to_string(),
            "--quants".to_string(),
            quants.to_string(),
            "--ram".to_string(),
            format!("{ram_gb:.1}"),
            "--cpu".to_string(),
            cpu.to_string(),
        ];
        self.start(python, engine_dir, ws, &args, Action::Feed);
    }

    fn open_profile(
        &mut self,
        python: &str,
        engine_dir: &Path,
        ws: &Path,
        author: &str,
        sort: &str,
    ) {
        if !self.profile_open {
            self.feed_backup = Some((
                std::mem::take(&mut self.results),
                std::mem::take(&mut self.sections),
            ));
            self.profile_open = true;
        }
        self.profile_author = Some(author.to_string());
        self.profile_sort = sort.to_string();
        self.reset_view();
        self.sections.clear();
        self.picker_open = false;
        self.auto_select = false;
        self.auto_searched = true;
        self.start(
            python,
            engine_dir,
            ws,
            &["author".into(), author.into(), sort.into()],
            Action::Profile,
        );
    }

    fn close_profile(&mut self) {
        if let Some((results, sections)) = self.feed_backup.take() {
            self.results = results;
            self.sections = sections;
        }
        self.profile_open = false;
        self.profile_author = None;
        self.profile_count = 0;
        self.selected = None;
        self.auto_select = true;
        self.auto_searched = true;
    }

    fn load_files(&mut self, python: &str, engine_dir: &Path, ws: &Path) {
        let Some(i) = self.selected else { return };
        let repo = self.results[i].id.clone();
        let kind = self.kind.clone();
        self.files.clear();
        self.variants.clear();
        self.file_sel = None;
        self.preview.clear();
        self.readme.clear();
        self.start(python, engine_dir, ws, &["files".into(), kind.clone(), repo.clone()], Action::Files);
        self.start(python, engine_dir, ws, &["readme".into(), kind, repo], Action::Readme);
    }

    pub fn download(&mut self, python: &str, engine_dir: &Path, ws: &Path, single: bool) {
        let Some(i) = self.selected else { return };
        let repo = self.results[i].id.clone();
        let kind = self.kind.clone();
        if !self.variants.is_empty() {
            if single {
                let Some(v) = self.file_sel.and_then(|k| self.variants.get(k)) else {
                    self.status = tr("Выбери квант").into();
                    return;
                };
                let label = v.label.clone();
                self.progress = Some((0, 0, String::new()));
                self.start(
                    python,
                    engine_dir,
                    ws,
                    &[
                        "download".into(),
                        kind,
                        repo,
                        "models_gguf".to_string(),
                        "--quant".into(),
                        label,
                    ],
                    Action::Download,
                );
                return;
            }
            self.progress = Some((0, 0, String::new()));
            self.start(
                python,
                engine_dir,
                ws,
                &["download".into(), kind, repo, "models_gguf".to_string()],
                Action::Download,
            );
            return;
        }
        let file = if single {
            match self.file_sel.and_then(|k| self.files.get(k)) {
                Some((name, _)) => Some(name.clone()),
                None => {
                    self.status = tr("Выбери файл в списке").into();
                    return;
                }
            }
        } else {
            None
        };
        let dest = match (&kind[..], file.as_deref()) {
            ("dataset", _) => format!("datasets/{}", repo.replace('/', "__")),
            (_, Some(f)) if f.to_lowercase().ends_with(".gguf") => "models_gguf".to_string(),
            _ => format!("models/{}", repo.replace('/', "__")),
        };
        self.progress = Some((0, 0, String::new()));
        let mut args = vec!["download".into(), kind, repo, dest];
        if let Some(f) = file {
            args.push(f);
        }
        self.start(python, engine_dir, ws, &args, Action::Download);
    }

    pub fn preview_selected(&mut self, python: &str, engine_dir: &Path, ws: &Path) {
        let Some(i) = self.selected else { return };
        let repo = self.results[i].id.clone();
        let kind = self.kind.clone();
        self.preview.clear();
        self.start(python, engine_dir, ws, &["preview".into(), kind, repo], Action::Preview);
    }

    pub fn poll(&mut self, python: &str, engine_dir: &Path, ws: &Path) {
        let Some((runner, action)) = self.proc.as_mut() else { return };
        let action: Action = *action;
        let lines = runner.drain();
        let finished = runner.poll_exit().is_some();
        for line in lines {
            self.handle_event(&line, action);
        }
        if finished {
            self.proc = None;
            self.busy = false;
            if let Some((args, action)) = self.queue.pop() {
                self.start(python, engine_dir, ws, &args, action);
            }
        }
    }

    /// Разбор одной строки hf_tools.py. Не-JSON уходит в журнал,
    /// JSON неизвестного вида молча пропускается (как раньше).
    fn handle_event(&mut self, line: &str, action: Action) {
        let event = match serde_json::from_str::<HubEvent>(line) {
            Ok(event) => event,
            Err(_) => {
                if serde_json::from_str::<Value>(line).is_err() {
                    self.log.push_back(line.to_string());
                }
                return;
            }
        };
        match event {
            HubEvent::Profile { count, author } => {
                self.profile_count = count;
                if let Some(a) = author {
                    self.profile_author = Some(a);
                }
            }
            HubEvent::Section { title, note } => {
                self.sections.push(Section {
                    title,
                    note,
                    start: self.results.len(),
                });
            }
            HubEvent::Item(p) => {
                self.results.push(Item {
                    id: p.id,
                    downloads: p.downloads,
                    likes: p.likes,
                    tags: p.tags,
                    author: p.author,
                    updated: p.updated,
                    created: p.created,
                    is_new: p.is_new,
                    library: p.library,
                    quant: p.quant,
                    file_size: p.file_size,
                    verified: p.verified,
                });
            }
            HubEvent::Files { items, variants } => {
                if let Some(items) = items {
                    self.files = items.into_iter().map(|f| (f.name, f.size)).collect();
                    self.variants = variants
                        .unwrap_or_default()
                        .into_iter()
                        .map(|v| Variant {
                            label: v.label,
                            size: v.size,
                        })
                        .collect();
                    self.file_sel = if !self.variants.is_empty() {
                        Some(self.default_variant())
                    } else if !self.files.is_empty() {
                        Some(0)
                    } else {
                        None
                    };
                    self.status = if self.variants.is_empty() {
                        crate::tf!("Файлов: {}", "Files: {}", self.files.len())
                    } else {
                        crate::tf!("Квантов: {}", "Quants: {}", self.variants.len())
                    };
                }
            }
            HubEvent::Progress { done, total, file } => {
                self.progress = Some((done, total, file));
            }
            HubEvent::Readme { text } => {
                if let Some(text) = text {
                    self.readme = text
                        .lines()
                        .map(|l| l.to_string())
                        .filter(|l| !l.trim_start().starts_with("---"))
                        .collect();
                }
            }
            HubEvent::Preview { rows, file } => {
                if let Some(rows) = rows {
                    self.preview.push(crate::tf!("файл: {file}", "file: {file}"));
                    for row in rows.iter().take(3) {
                        let short = serde_json::to_string_pretty(row).unwrap_or_default();
                        self.preview.push(short.chars().take(600).collect());
                    }
                }
            }
            HubEvent::Done => {
                self.status = match action {
                    Action::Search => crate::tf!("Найдено: {}", "Found: {}", self.results.len()),
                    Action::Feed => crate::tf!("В ленте: {}", "In feed: {}", self.results.len()),
                    Action::Profile => crate::tf!("Моделей: {}", "Models: {}", self.results.len()),
                    Action::Files => crate::tf!("Файлов: {}", "Files: {}", self.files.len()),
                    Action::Download => tr("Скачано").into(),
                    Action::Preview => tr("Готово").into(),
                    Action::Readme => tr("README загружен").into(),
                };
            }
            HubEvent::Error { message } => {
                self.status = message.unwrap_or_else(|| tr("ошибка").to_string());
            }
            HubEvent::Unknown => {}
        }
    }


    /// Синяя галочка рядом с ником (как на Hugging Face).
    fn verified_badge(painter: &egui::Painter, center: egui::Pos2) {
        const BLUE: egui::Color32 = egui::Color32::from_rgb(0x2F, 0x80, 0xED);
        painter.circle_filled(center, 5.5, BLUE);
        let s = egui::Stroke::new(1.3_f32, egui::Color32::WHITE);
        painter.line_segment(
            [
                egui::pos2(center.x - 2.5, center.y),
                egui::pos2(center.x - 0.5, center.y + 2.0),
            ],
            s,
        );
        painter.line_segment(
            [
                egui::pos2(center.x - 0.5, center.y + 2.0),
                egui::pos2(center.x + 2.8, center.y - 2.2),
            ],
            s,
        );
    }

    fn short_num(v: u64) -> String {
        if v >= 1_000_000 {
            format!("{:.1}M", v as f64 / 1e6)
        } else if v >= 1_000 {
            format!("{:.1}K", v as f64 / 1e3)
        } else {
            v.to_string()
        }
    }

    fn tile_color(name: &str) -> egui::Color32 {
        let sum: u32 = name.bytes().map(|b| b as u32).sum();
        const COLORS: [egui::Color32; 6] = [
            egui::Color32::from_rgb(0x0E, 0x9F, 0x6E),
            egui::Color32::from_rgb(0x3B, 0x82, 0xF6),
            egui::Color32::from_rgb(0x8B, 0x5C, 0xF6),
            egui::Color32::from_rgb(0xF5, 0x9E, 0x0B),
            egui::Color32::from_rgb(0xEF, 0x44, 0x44),
            egui::Color32::from_rgb(0x14, 0xB8, 0xA6),
        ];
        COLORS[(sum as usize) % COLORS.len()]
    }

    fn logo_tile(ui: &mut egui::Ui, id: &str, size: f32) {
        let name = id.split('/').nth(1).unwrap_or(id);
        let letter = name
            .chars()
            .find(|c| c.is_alphanumeric())
            .unwrap_or('?')
            .to_uppercase()
            .to_string();
        let (rect, _) = ui.allocate_exact_size(egui::vec2(size, size), egui::Sense::hover());
        let painter = ui.painter();
        painter.rect_filled(
            rect,
            egui::Rounding::same(size * 0.24),
            Self::tile_color(id),
        );
        painter.text(
            rect.center(),
            egui::Align2::CENTER_CENTER,
            letter,
            egui::FontId::new(size * 0.44, theme::bold_family()),
            egui::Color32::WHITE,
        );
    }

    pub fn draw(
        &mut self,
        ui: &mut egui::Ui,
        theme: &Theme,
        python: &str,
        engine_dir: &Path,
        ws: &Path,
        local: &[ModelEntry],
        cpu: u32,
        ram_gb: f64,
        ram_total: f64,
        vram_gb: f64,
        hf_user: &str,
        quant_mode: &str,
    ) -> Option<HubAction> {
        let p = theme.pal;
        let mut action = None;
        self.ram_gb = ram_gb as f32;
        self.ram_total = ram_total;
        self.cpu = cpu;
        if ui.input(|i| i.key_pressed(egui::Key::Escape))
            && !self.picker_open
            && self.profile_open
        {
            if self.selected.is_some() {
                self.selected = None;
                self.files.clear();
                self.variants.clear();
                self.file_sel = None;
                self.readme.clear();
            } else {
                self.close_profile();
            }
        }
        if self.feed_user != hf_user || self.feed_quants != quant_mode {
            self.feed_user = hf_user.to_string();
            self.feed_quants = quant_mode.to_string();
            if !self.feed_user.is_empty() {
                self.feed_loaded = false;
                self.auto_searched = false;
                if self.kind == "model" && self.query.trim().is_empty() {
                    self.results.clear();
                    self.sections.clear();
                    self.selected = None;
                }
            }
        }
        if !self.on_device && !self.auto_searched && !self.busy && self.results.is_empty() {
            self.auto_searched = true;
            if self.kind == "model" && self.query.trim().is_empty() {
                self.load_feed(python, engine_dir, ws, hf_user, quant_mode, self.ram_gb, self.cpu);
            } else {
                self.search(python, engine_dir, ws);
            }
        }
        // Как в Unsloth: первый результат сразу открыт в карточке.
        if self.auto_select && !self.busy && self.selected.is_none() && !self.results.is_empty() {
            self.auto_select = false;
            self.selected = Some(0);
            self.load_files(python, engine_dir, ws);
        }

        ui.label(theme.overline(tr("Каталог")));
        ui.add_space(2.0);
        ui.label(theme.display(tr("Хаб моделей"), 36.0));
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            ui.label(theme.label(tr("Ищи и скачивай модели и датасеты, выбирай квант")));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                theme::chip(ui, theme, &crate::tf!("{} локальных", "{} local", local.len()));
                if vram_gb > 0.1 {
                    theme::chip(ui, theme, &crate::tf!("VRAM {:.0} ГБ", "VRAM {:.0} GB", vram_gb));
                }
                theme::chip(ui, theme, &crate::tf!("RAM {:.1} / {:.0} ГБ", "RAM {:.1} / {:.0} GB", ram_gb, ram_total));
                theme::chip(ui, theme, &format!("CPU {}", cpu));
            });
        });
        ui.add_space(6.0);
        theme.hairline(ui);
        ui.add_space(10.0);

        ui.horizontal(|ui| {
            if self.profile_open {
                let author = self.profile_author.clone().unwrap_or_default();
                let (rect, resp) =
                    ui.allocate_exact_size(egui::vec2(26.0, 26.0), egui::Sense::click());
                if resp.hovered() {
                    ui.painter().circle_filled(
                        rect.center(),
                        13.0,
                        p.surface_container_high,
                    );
                }
                theme::draw_icon(
                    ui.painter(),
                    rect.center(),
                    theme::icons::ARROW_BACK,
                    18.0,
                    p.on_surface_variant,
                );
                if resp.clicked() {
                    self.close_profile();
                }
                ui.label(
                    RichText::new(&author)
                        .font(theme::title_font(18.0))
                        .color(p.on_surface),
                );
                let verified = author.eq_ignore_ascii_case(hf_user)
                    || author.eq_ignore_ascii_case("unsloth");
                if verified {
                    Self::verified_ui(ui);
                }
                if self.busy {
                    ui.add(egui::Spinner::new().size(16.0).color(p.primary));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let cur = if self.profile_sort == "createdAt" {
                        tr("Новые")
                    } else {
                        tr("Популярные")
                    };
                    if let Some(picked) =
                        theme::segmented(ui, theme, &[tr("Популярные"), tr("Новые")], cur)
                    {
                        let sort = if picked == tr("Новые") {
                            "createdAt"
                        } else {
                            "downloads"
                        };
                        if sort != self.profile_sort {
                            self.profile_sort = sort.to_string();
                            self.open_profile(python, engine_dir, ws, &author, sort);
                        }
                    }
                });
            } else {
                if let Some(v) = theme::segmented(ui, theme, &["Discover", "On Device"], if self.on_device { "On Device" } else { "Discover" }) {
                    self.on_device = v == "On Device";
                }
                if !self.on_device {
                    if let Some(k) = theme::segmented(ui, theme, &["model", "dataset"], &self.kind.clone()) {
                        if k != self.kind {
                            if self.profile_open {
                                self.close_profile();
                            }
                            self.kind = k;
                            self.results.clear();
                            self.sections.clear();
                            self.selected = None;
                            self.files.clear();
                            self.auto_searched = false;
                            if self.kind == "model" && self.query.trim().is_empty() {
                                self.feed_loaded = false;
                            }
                        }
                    }
                }
                if self.busy {
                    ui.add(egui::Spinner::new().size(16.0).color(p.primary));
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let cur = if self.sort == "createdAt" { tr("Новые") } else { tr("Популярные") };
                    if !self.query.trim().is_empty() {
                        if let Some(picked) =
                            theme::segmented(ui, theme, &[tr("Популярные"), tr("Новые")], cur)
                        {
                            let sort = if picked == tr("Новые") { "createdAt" } else { "downloads" };
                            if sort != self.sort {
                                self.sort = sort.to_string();
                                self.search(python, engine_dir, ws);
                            }
                        }
                    }
                });
            }
        });
        ui.add_space(8.0);

        if self.on_device {
            self.draw_on_device(ui, theme, local, &mut action);
            return action;
        }

        if !self.profile_open {
        ui.horizontal(|ui| {
            // M3 search bar: тональная капсула 28dp с лупой.
            let bar_w = (ui.available_width() - 104.0).max(200.0);
            egui::Frame::none()
                .fill(p.surface_container_high)
                .rounding(egui::Rounding::same(4.0))
                .inner_margin(egui::Margin::symmetric(14.0, 9.0))
                .show(ui, |ui| {
                    ui.set_width(bar_w);
                    ui.horizontal(|ui| {
                        let (rect, _) = ui.allocate_exact_size(
                            egui::vec2(16.0, 16.0),
                            egui::Sense::hover(),
                        );
                        theme::draw_icon(
                            ui.painter(),
                            rect.center(),
                            theme::icons::SEARCH,
                            16.0,
                            p.on_surface_variant,
                        );
                        ui.add_space(4.0);
                        ui.add(
                            egui::TextEdit::singleline(&mut self.query)
                                .frame(false)
                                .hint_text(tr("Поиск моделей и датасетов…"))
                                .desired_width(f32::INFINITY),
                        );
                    });
                });
            if theme::button(ui, theme, tr("Найти"), ButtonKind::Filled, !self.busy).clicked() {
                self.search(python, engine_dir, ws);
            }
        });
        if self.busy && !self.status.is_empty() {
            ui.label(theme.label(&self.status));
        }
        ui.add_space(8.0);
        }

        let content_h = ui.available_height().max(220.0);
        let width = ui.available_width();
        let left_w = (width * 0.42).max(280.0);
        let right_w = (width - left_w - 16.0).max(300.0);
        ui.horizontal_top(|ui| {
            ui.vertical(|ui| {
                ui.set_width(left_w);
                ScrollArea::vertical()
                    .id_salt("hub_results")
                    .max_height(content_h)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                    ui.set_width((left_w - 16.0).max(200.0));
                    if self.profile_open {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(tr("Модели автора"))
                                    .font(theme::title_font(14.0))
                                    .color(p.on_surface),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.label(theme.mono(
                                        format!("{}", self.results.len()),
                                        10.0,
                                    ));
                                },
                            );
                        });
                        ui.add_space(4.0);
                        theme.hairline(ui);
                    }
                    if self.results.is_empty() {
                        ui.label(theme.label(tr("Результаты появятся здесь.")));
                    }
                    let mut bounds: Vec<(String, String, usize, usize)> = Vec::new();
                    if self.sections.is_empty() {
                        bounds.push((String::new(), String::new(), 0, self.results.len()));
                    } else {
                        for (n, sec) in self.sections.iter().enumerate() {
                            let end = self
                                .sections
                                .get(n + 1)
                                .map(|s| s.start)
                                .unwrap_or(self.results.len());
                            if sec.start < end {
                                bounds.push((sec.title.clone(), sec.note.clone(), sec.start, end));
                            }
                        }
                    }
                    for (title, note, from, to) in bounds {
                        if !title.is_empty() {
                            ui.add_space(12.0);
                            ui.horizontal(|ui| {
                                ui.label(
                                    RichText::new(&title)
                                        .font(theme::title_font(14.0))
                                        .color(p.on_surface),
                                );
                                if !note.is_empty() {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            ui.label(theme.mono(note.to_uppercase(), 9.5));
                                        },
                                    );
                                }
                            });
                            ui.add_space(4.0);
                            theme.hairline(ui);
                        }
                        for i in from..to {
                            let resp = self.result_row(ui, theme, i, hf_user);
                            if resp.clicked() {
                                self.selected = Some(i);
                                self.load_files(python, engine_dir, ws);
                            }
                        }
                    }
                    });
            });
            ui.add_space(8.0);
            ui.vertical(|ui| {
                ui.set_width((right_w - 8.0).max(240.0));
                if self.profile_open && self.selected.is_none() {
                    let author = self.profile_author.clone().unwrap_or_default();
                    Self::draw_profile(ui, theme, &author, &self.results, hf_user);
                } else {
                    let details_id = ("hub_details", self.selected);
                    ScrollArea::vertical()
                        .id_salt(details_id)
                        .max_height(content_h)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            self.draw_details(ui, theme, python, engine_dir, ws, &mut action, hf_user);
                        });
                    if let Some(a) = self.pending_profile.take() {
                        self.open_profile(python, engine_dir, ws, &a, "downloads");
                    }
                }
            });
        });
        action
    }

    /// Строка результата: логотип, имя, автор с галочкой, счётчики и дата.
    fn result_row(
        &self,
        ui: &mut egui::Ui,
        theme: &Theme,
        i: usize,
        hf_user: &str,
    ) -> egui::Response {
        let p = theme.pal;
        let Some(item) = self.results.get(i) else {
            return ui.allocate_response(egui::vec2(0.0, 0.0), egui::Sense::hover());
        };
        let selected = self.selected == Some(i);
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 52.0),
            egui::Sense::click(),
        );
        let fill = if selected {
            p.surface_container_high
        } else if resp.hovered() {
            p.surface_container_low
        } else {
            egui::Color32::TRANSPARENT
        };
        let painter = ui.painter().clone();
        painter.rect_filled(rect, egui::Rounding::same(4.0), fill);
        if selected {
            painter.rect_filled(
                egui::Rect::from_min_size(
                    egui::pos2(rect.left() + 2.0, rect.center().y - 14.0),
                    egui::vec2(3.0, 28.0),
                ),
                egui::Rounding::ZERO,
                p.primary,
            );
        }
        let name = item.id.split('/').nth(1).unwrap_or(&item.id);
        let tile = egui::Rect::from_min_size(
            egui::pos2(rect.left() + 10.0, rect.center().y - 14.0),
            egui::vec2(28.0, 28.0),
        );
        painter.rect_filled(tile, egui::Rounding::same(8.0), Self::tile_color(&item.id));
        painter.text(
            tile.center(),
            egui::Align2::CENTER_CENTER,
            name.chars().next().unwrap_or('?').to_uppercase().to_string(),
            egui::FontId::new(13.0, theme::bold_family()),
            egui::Color32::WHITE,
        );
        let text_x = rect.left() + 48.0;
        let name_font = theme::title_font(13.0);
        let max_name_w = (rect.width() - 220.0).max(80.0);
        let mut shown = name.to_string();
        let mut name_w = ui.fonts(|f| {
            f.layout_no_wrap(shown.clone(), name_font.clone(), p.on_surface)
                .size()
                .x
        });
        while name_w > max_name_w && shown.chars().count() > 5 {
            let keep = shown.chars().count().saturating_sub(2);
            shown = format!("{}…", shown.chars().take(keep).collect::<String>());
            name_w = ui.fonts(|f| {
                f.layout_no_wrap(shown.clone(), name_font.clone(), p.on_surface)
                    .size()
                    .x
            });
        }
        painter.text(
            egui::pos2(text_x, rect.center().y - 9.0),
            egui::Align2::LEFT_CENTER,
            &shown,
            name_font.clone(),
            p.on_surface,
        );
        if item.is_new {
            painter.circle_filled(
                egui::pos2(text_x + name_w + 8.0, rect.center().y - 9.0),
                3.0,
                p.primary,
            );
        }
        let verified = item.verified
            || item.author.eq_ignore_ascii_case(hf_user)
            || item.author.eq_ignore_ascii_case("unsloth");
        let author = if item.author.is_empty() {
            "Hugging Face".to_string()
        } else {
            item.author.clone()
        };
        let author_font = egui::FontId::new(10.5, egui::FontFamily::Monospace);
        painter.text(
            egui::pos2(text_x, rect.center().y + 9.0),
            egui::Align2::LEFT_CENTER,
            &author,
            author_font.clone(),
            p.on_surface_variant,
        );
        let author_w = ui.fonts(|f| {
            f.layout_no_wrap(author.clone(), author_font.clone(), p.on_surface_variant)
                .size()
                .x
        });
        let mut meta_x = text_x + author_w + 8.0;
        if verified {
            Self::verified_badge(&painter, egui::pos2(meta_x + 5.0, rect.center().y + 9.0));
            meta_x += 16.0;
        }
        if !item.quant.is_empty() {
            let meta = if item.file_size > 0 {
                crate::tf!("{} · {:.1} ГБ", "{} · {:.1} GB", item.quant, item.file_size as f64 / 1e9)
            } else {
                item.quant.clone()
            };
            painter.text(
                egui::pos2(meta_x, rect.center().y + 9.0),
                egui::Align2::LEFT_CENTER,
                meta,
                author_font.clone(),
                p.on_surface_variant,
            );
        }
        let right = rect.right() - 10.0;
        let small = egui::FontId::new(10.5, egui::FontFamily::Monospace);
        let dl = Self::short_num(item.downloads);
        let dl_w = ui.fonts(|f| {
            f.layout_no_wrap(dl.clone(), small.clone(), p.on_surface_variant)
                .size()
                .x
        });
        // Числа прижаты к фиксированным правым краям, чтобы столбцы не гуляли.
        painter.text(
            egui::pos2(right, rect.center().y - 9.0),
            egui::Align2::RIGHT_CENTER,
            &dl,
            small.clone(),
            p.on_surface_variant,
        );
        let icon_center = egui::pos2(right - dl_w - 8.0, rect.center().y - 9.0);
        theme::draw_icon(
            &painter,
            icon_center,
            theme::icons::DOWNLOAD,
            13.0,
            p.on_surface_variant,
        );
        painter.text(
            egui::pos2(right - 86.0, rect.center().y - 9.0),
            egui::Align2::RIGHT_CENTER,
            format!("♥ {}", item.likes),
            small,
            p.on_surface_variant,
        );
        if !item.updated.is_empty() {
            painter.text(
                egui::pos2(right, rect.center().y + 9.0),
                egui::Align2::RIGHT_CENTER,
                &item.updated,
                egui::FontId::new(10.0, egui::FontFamily::Monospace),
                p.on_surface_variant,
            );
        }
        painter.line_segment(
            [
                egui::pos2(rect.left() + 10.0, rect.bottom()),
                egui::pos2(rect.right() - 10.0, rect.bottom()),
            ],
            egui::Stroke::new(1.0_f32, p.outline_variant),
        );
        resp.on_hover_text(&item.id)
            .on_hover_cursor(egui::CursorIcon::PointingHand)
    }

    fn pretty_tag(tag: &str) -> Option<String> {
        let t = tag.to_lowercase();
        if t.contains("endpoints_compatible")
            || t.starts_with("region:")
            || t.contains("autotrain")
        {
            return None;
        }
        let label = if t.contains("conversational") {
            tr("Диалоги")
        } else if t.contains("text-generation") {
            tr("Текст")
        } else if t.contains("multilingual") {
            tr("Мультиязычная")
        } else if t.contains("code") {
            tr("Код")
        } else if t == "gguf" {
            "GGUF"
        } else if t.contains("vision") || t.contains("image") {
            tr("Изображения")
        } else if t.contains("audio") {
            tr("Аудио")
        } else if t.starts_with("base_model:") {
            tr("База")
        } else {
            tag
        };
        Some(label.chars().take(20).collect())
    }

    fn tag_color(tag: &str) -> egui::Color32 {
        let t = tag.to_lowercase();
        let c = if t.contains("conversational") {
            (0xB3, 0x88, 0xFF)
        } else if t.contains("multilingual") || t == "gguf" {
            (0x64, 0xB5, 0xF6)
        } else if t.contains("code") {
            (0x4A, 0xDE, 0x80)
        } else if t.contains("text-generation") {
            (0x2D, 0xD4, 0xBF)
        } else if t.contains("vision") || t.contains("image") {
            (0xF4, 0x72, 0xB6)
        } else if t.contains("audio") {
            (0xFB, 0xBF, 0x24)
        } else {
            (0x9C, 0xA3, 0xAF)
        };
        egui::Color32::from_rgb(c.0, c.1, c.2)
    }

    fn verified_ui(ui: &mut egui::Ui) {
        let (rect, _) =
            ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
        Self::verified_badge(ui.painter(), rect.center());
    }

    /// Простой рендер markdown: заголовки, списки, ссылки, код.
    fn draw_readme(ui: &mut egui::Ui, theme: &Theme, lines: &[String]) {
        let p = theme.pal;
        let mut code = false;
        for raw in lines {
            let line = raw.trim_end();
            let trim = line.trim();
            if trim.starts_with("```") {
                code = !code;
                continue;
            }
            if trim.is_empty() {
                ui.add_space(4.0);
                continue;
            }
            if code {
                ui.label(
                    RichText::new(trim)
                        .monospace()
                        .size(10.5)
                        .color(p.on_surface_variant),
                );
                continue;
            }
            if trim.starts_with("![") || trim.starts_with("[!") {
                continue;
            }
            let level = trim.chars().take_while(|c| *c == '#').count();
            if level > 0 {
                let text = trim[level..].trim();
                let size = match level {
                    1 => 17.0,
                    2 => 14.5,
                    _ => 12.5,
                };
                ui.add_space(3.0);
                ui.label(
                    RichText::new(Self::clean_inline(text))
                        .size(size)
                        .color(p.on_surface)
                        .strong(),
                );
                continue;
            }
            let bullet = trim.starts_with("- ") || trim.starts_with("* ");
            let text = if bullet { trim[2..].trim() } else { trim };
            if text.starts_with('|') {
                let row = text.trim_matches('|').replace('|', "   ");
                ui.label(
                    RichText::new(Self::clean_inline(&row))
                        .size(10.5)
                        .color(p.on_surface_variant),
                );
                continue;
            }
            ui.horizontal_wrapped(|ui| {
                if bullet {
                    ui.label(
                        RichText::new("•")
                            .size(11.5)
                            .color(p.on_surface_variant),
                    );
                }
                Self::md_runs(ui, theme, text, 11.5);
            });
        }
    }

    fn clean_inline(text: &str) -> String {
        let flat = text.replace("**", "").replace("__", "").replace('`', "");
        // README часто начинается с эмодзи, которых нет в Roboto — вырезаем,
        // чтобы вместо них не рисовались пустые квадраты.
        flat.chars()
            .filter(|c| {
                let u = *c as u32;
                !matches!(u,
                    0x200D | 0x20E3 | 0xFE0F
                    | 0x1F000..=0x1FAFF
                    | 0x2600..=0x27BF
                    | 0x2B00..=0x2BFF
                    | 0x25A0..=0x25FF)
            })
            .collect()
    }

    fn md_runs(ui: &mut egui::Ui, theme: &Theme, text: &str, size: f32) {
        let p = theme.pal;
        let mut rest = text;
        loop {
            let link = rest.find('[');
            let Some(start) = link else { break };
            let Some(mid_rel) = rest[start..].find("](") else { break };
            let mid = start + mid_rel;
            let Some(end_rel) = rest[mid + 2..].find(')') else { break };
            let end = mid + 2 + end_rel;
            let label = &rest[start + 1..mid];
            let url = &rest[mid + 2..end];
            let prefix = Self::clean_inline(&rest[..start]);
            if !prefix.trim().is_empty() {
                ui.label(
                    RichText::new(prefix.trim())
                        .size(size)
                        .color(p.on_surface_variant),
                );
            }
            if url.starts_with("http") {
                ui.hyperlink_to(
                    RichText::new(Self::clean_inline(label))
                        .size(size)
                        .color(p.primary),
                    url,
                );
            } else {
                ui.label(
                    RichText::new(Self::clean_inline(label))
                        .size(size)
                        .color(p.on_surface_variant),
                );
            }
            rest = &rest[end + 1..];
        }
        let tail = Self::clean_inline(rest);
        if !tail.trim().is_empty() {
            ui.label(
                RichText::new(tail.trim())
                    .size(size)
                    .color(p.on_surface_variant),
            );
        }
    }

    fn fmt_weight(size: u64) -> String {
        let gb = size as f64 / 1e9;
        if gb >= 100.0 {
            crate::tf!("{gb:.0} ГБ", "{gb:.0} GB")
        } else if gb >= 10.0 {
            crate::tf!("{gb:.1} ГБ", "{gb:.1} GB")
        } else {
            crate::tf!("{gb:.2} ГБ", "{gb:.2} GB")
        }
    }

    /// Селектор кванта как в Unsloth: крупная строка + список с весами.
    fn quant_selector(&mut self, ui: &mut egui::Ui, theme: &Theme, width: f32) {
        let p = theme.pal;
        let variants: Vec<(String, u64)> = self
            .variants
            .iter()
            .map(|v| (v.label.clone(), v.size))
            .collect();
        if variants.is_empty() {
            return;
        }
        let sel = self
            .file_sel
            .unwrap_or(0)
            .min(variants.len().saturating_sub(1));
        let (rect, resp) = ui.allocate_exact_size(
            egui::vec2(width, 46.0),
            egui::Sense::click(),
        );
        let painter = ui.painter().clone();
        let fill = if resp.hovered() || self.picker_open {
            p.surface_container_high
        } else {
            p.surface_container
        };
        painter.rect_filled(rect, egui::Rounding::same(12.0), fill);

        let center = rect.center();
        painter.circle_stroke(
            egui::pos2(rect.left() + 22.0, center.y),
            7.0,
            egui::Stroke::new(1.2_f32, p.on_surface_variant),
        );
        painter.text(
            egui::pos2(rect.left() + 22.0, center.y),
            egui::Align2::CENTER_CENTER,
            "i",
            egui::FontId::new(9.0, theme::medium_family()),
            p.on_surface_variant,
        );
        painter.text(
            egui::pos2(rect.left() + 38.0, center.y),
            egui::Align2::LEFT_CENTER,
            &variants[sel].0,
            theme::title_font(13.0),
            p.on_surface,
        );
        let label_w = ui.fonts(|f| {
            f.layout_no_wrap(
                variants[sel].0.clone(),
                theme::title_font(13.0),
                p.on_surface,
            )
            .size()
            .x
        });
        let tag_x = rect.left() + 46.0 + label_w;
        painter.circle_filled(
            egui::pos2(tag_x + 3.0, center.y),
            3.0,
            egui::Color32::from_rgb(0x4C, 0x9A, 0xFF),
        );
        painter.text(
            egui::pos2(tag_x + 10.0, center.y),
            egui::Align2::LEFT_CENTER,
            "GGUF",
            egui::FontId::new(10.0, egui::FontFamily::Monospace),
            p.on_surface_variant,
        );
        painter.text(
            egui::pos2(rect.right() - 30.0, center.y),
            egui::Align2::RIGHT_CENTER,
            Self::fmt_weight(variants[sel].1),
            egui::FontId::new(11.0, egui::FontFamily::Monospace),
            p.on_surface,
        );
        theme::draw_icon(
            &painter,
            egui::pos2(rect.right() - 16.0, center.y),
            theme::icons::EXPAND_MORE,
            18.0,
            p.on_surface_variant,
        );
        if resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked() {
            self.picker_open = !self.picker_open;
            self.picker_pos = Some(rect.left_bottom() + egui::vec2(0.0, 4.0));
            self.picker_width = rect.width();
        }
        if self.picker_open {
            self.quant_popup(ui.ctx(), theme);
        }
    }

    fn quant_popup(&mut self, ctx: &egui::Context, theme: &Theme) {
        let Some(pos) = self.picker_pos else { return };
        let p = theme.pal;
        let variants: Vec<(String, u64)> = self
            .variants
            .iter()
            .map(|v| (v.label.clone(), v.size))
            .collect();
        let sel = self.file_sel.unwrap_or(0);
        let width = self.picker_width.max(240.0);
        let mut close = false;

        egui::Area::new("hub_quant_scrim".into())
            .fixed_pos(egui::pos2(0.0, 0.0))
            .order(egui::Order::Middle)
            .show(ctx, |ui| {
                let (rect, resp) = ui.allocate_exact_size(
                    ctx.screen_rect().size(),
                    egui::Sense::click(),
                );
                ui.painter().rect_filled(rect, 0.0, egui::Color32::TRANSPARENT);
                if resp.clicked() {
                    close = true;
                }
            });
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            close = true;
        }

        let mut picked = None;
        egui::Area::new("hub_quant_popup".into())
            .fixed_pos(pos)
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::none()
                    .fill(p.surface_container_high)
                    .rounding(egui::Rounding::same(12.0))
                    .inner_margin(egui::Margin::same(6.0))
                    .shadow(egui::epaint::Shadow {
                        offset: egui::vec2(0.0, 8.0),
                        blur: 24.0,
                        spread: 0.0,
                        color: egui::Color32::from_black_alpha(90),
                    })
                    .show(ui, |ui| {
                        ui.set_width(width);
                        egui::ScrollArea::vertical()
                            .id_salt("hub_quant_list")
                            .max_height(330.0)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.set_width(width);
                                for (i, (label, size)) in variants.iter().enumerate() {
                                    let (r, resp) = ui.allocate_exact_size(
                                        egui::vec2(width - 12.0, 44.0),
                                        egui::Sense::click(),
                                    );
                                    let fill = if i == sel {
                                        p.surface_container_highest
                                    } else if resp.hovered() {
                                        p.surface_container
                                    } else {
                                        egui::Color32::TRANSPARENT
                                    };
                                    let painter = ui.painter().clone();
                                    painter.rect_filled(r, egui::Rounding::same(8.0), fill);
                                    painter.text(
                                        egui::pos2(r.left() + 14.0, r.center().y),
                                        egui::Align2::LEFT_CENTER,
                                        label,
                                        theme::title_font(13.0),
                                        p.on_surface,
                                    );
                                    painter.text(
                                        egui::pos2(r.right() - 14.0, r.center().y),
                                        egui::Align2::RIGHT_CENTER,
                                        Self::fmt_weight(*size),
                                        egui::FontId::new(11.0, egui::FontFamily::Monospace),
                                        p.on_surface_variant,
                                    );
                                    if resp.clicked() {
                                        picked = Some(i);
                                    }
                                }
                            });
                    });
            });
        if let Some(i) = picked {
            self.file_sel = Some(i);
            close = true;
        }
        if close {
            self.picker_open = false;
        }
    }

    fn draw_profile(
        ui: &mut egui::Ui,
        theme: &Theme,
        author: &str,
        items: &[Item],
        hf_user: &str,
    ) {
        let p = theme.pal;
        let downloads: u64 = items.iter().map(|i| i.downloads).sum();
        let likes: u64 = items.iter().map(|i| i.likes).sum();
        let verified = author.eq_ignore_ascii_case(hf_user)
            || author.eq_ignore_ascii_case("unsloth");
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            Self::logo_tile(ui, &format!("{author}/{author}"), 56.0);
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(theme.display(author, 28.0));
                    if verified {
                        Self::verified_ui(ui);
                    }
                });
                ui.label(
                    RichText::new(crate::tf!("{} GGUF-моделей · Σ {} · ♥ {}", "{} GGUF models · Σ {} · ♥ {}",
                        items.len(),
                        Self::short_num(downloads),
                        likes
                    ))
                    .font(egui::FontId::new(10.5, egui::FontFamily::Monospace))
                    .color(p.on_surface_variant),
                );
            });
        });
        ui.add_space(10.0);
        theme.hairline(ui);
        ui.add_space(10.0);
        if items.is_empty() {
            ui.add(egui::Spinner::new().size(16.0).color(p.primary));
            ui.add_space(4.0);
            ui.label(theme.label(tr("Загружаю модели автора…")));
        } else {
            ui.label(theme.label(
                tr("Слева — все GGUF-модели автора. Открой любую, чтобы посмотреть кванты."),
            ));
            ui.add_space(4.0);
            ui.label(theme.mono(tr("Esc — вернуться к ленте"), 10.5));
        }
    }

    fn draw_details(
        &mut self,
        ui: &mut egui::Ui,
        theme: &Theme,
        python: &str,
        engine_dir: &Path,
        ws: &Path,
        action: &mut Option<HubAction>,
        hf_user: &str,
    ) {
        let p = theme.pal;
        theme.card(p.surface_container_lowest, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            let Some(i) = self.selected else {
                ui.label(theme.label(tr("Выбери модель слева, чтобы увидеть файлы и скачать.")));
                return;
            };
            let id = self.results[i].id.clone();
            let tags = self.results[i].tags.clone();
            let downloads = self.results[i].downloads;
            let likes = self.results[i].likes;
            let item_updated = self.results[i].updated.clone();
            let item_created = self.results[i].created.clone();
            let item_library = self.results[i].library.clone();
            let item_new = self.results[i].is_new;

            let (author, name) = id.split_once('/').unwrap_or(("", id.as_str()));
            let verified = author.eq_ignore_ascii_case(hf_user)
                || author.eq_ignore_ascii_case("unsloth");
            ui.horizontal(|ui| {
                Self::logo_tile(ui, &id, 46.0);
                ui.vertical(|ui| {
                    ui.label(
                        RichText::new(name)
                            .font(theme::title_font(22.0))
                            .color(p.on_surface),
                    );
                    ui.horizontal(|ui| {
                        let author_text = if author.is_empty() {
                            "Hugging Face".to_string()
                        } else {
                            author.to_string()
                        };
                        let font = egui::FontId::new(10.5, egui::FontFamily::Monospace);
                        let galley = ui.painter().layout_no_wrap(
                            author_text.clone(),
                            font,
                            p.primary,
                        );
                        let (lrect, lresp) = ui.allocate_exact_size(
                            galley.size() + egui::vec2(6.0, 4.0),
                            egui::Sense::click(),
                        );
                        if lresp.hovered() {
                            ui.painter().rect_filled(
                                lrect,
                                egui::Rounding::same(4.0),
                                p.surface_container_high,
                            );
                        }
                        ui.painter().galley(
                            lrect.min + egui::vec2(3.0, 2.0),
                            galley,
                            p.primary,
                        );
                        if lresp.clicked() && !author.is_empty() {
                            self.pending_profile = Some(author.to_string());
                        }
                        lresp.on_hover_cursor(egui::CursorIcon::PointingHand);
                        if verified {
                            Self::verified_ui(ui);
                        }
                        if item_new {
                            ui.label(theme.overline(tr("новое")));
                        }
                    });
                    ui.add_space(4.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                        for t in tags.iter().take(4) {
                            if let Some(label) = Self::pretty_tag(t) {
                                theme::tag_chip(ui, theme, &label, Self::tag_color(t));
                            }
                        }
                    });
                    ui.add_space(2.0);
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = egui::vec2(4.0, 4.0);
                        if !item_updated.is_empty() {
                            theme::chip(ui, theme, &crate::tf!("обновлено {}", "updated {}", item_updated));
                        }
                        if !item_created.is_empty() {
                            theme::chip(ui, theme, &crate::tf!("создано {}", "created {}", item_created));
                        }
                        theme::chip(ui, theme, &format!("♥ {}", likes));
                        theme::icon_chip(ui, theme, theme::icons::DOWNLOAD, &Self::short_num(downloads));
                        if !item_library.is_empty()
                            && !item_library.eq_ignore_ascii_case("gguf")
                        {
                            theme::chip(ui, theme, &item_library);
                        }
                    });
                });
            });
            ui.add_space(10.0);

            let current = self
                .file_sel
                .and_then(|k| self.files.get(k))
                .map(|(n, s)| crate::tf!("{}   ·   {:.2} ГБ", "{}   ·   {:.2} GB", n, *s as f64 / 1e9))
                .unwrap_or_else(|| {
                    if self.files.is_empty() {
                        tr("читаю список файлов…").to_string()
                    } else {
                        tr("выбери файл").to_string()
                    }
                });
            ui.horizontal(|ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if theme::button(ui, theme, tr("Скачать"), ButtonKind::Filled, !self.busy).clicked() {
                        self.download(python, engine_dir, ws, true);
                    }
                    if !self.variants.is_empty() {
                        let width = ui.available_width();
                        self.quant_selector(ui, theme, width);
                    } else {
                        egui::Frame::none()
                            .fill(p.surface_container_highest)
                            .rounding(egui::Rounding::same(4.0))
                            .inner_margin(egui::Margin::symmetric(12.0, 8.0))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                egui::ComboBox::from_id_salt("hub_file")
                                    .selected_text(RichText::new(current).size(12.5))
                                    .width(ui.available_width())
                                    .show_ui(ui, |ui| {
                                        for k in 0..self.files.len() {
                                            let (n, s) = self.files[k].clone();
                                            ui.selectable_value(
                                                &mut self.file_sel,
                                                Some(k),
                                                crate::tf!("{}   ·   {:.2} ГБ", "{}   ·   {:.2} GB", n, s as f64 / 1e9),
                                            );
                                        }
                                    });
                            });
                    }
                });
            });
            ui.add_space(6.0);
            ui.horizontal_wrapped(|ui| {
                if self.variants.is_empty()
                    && theme::button(ui, theme, tr("Скачать все файлы"), ButtonKind::Tonal, !self.busy).clicked()
                {
                    self.download(python, engine_dir, ws, false);
                }
                if self.kind == "dataset"
                    && theme::button(ui, theme, tr("Предпросмотр"), ButtonKind::Tonal, !self.busy).clicked()
                {
                    self.preview_selected(python, engine_dir, ws);
                }
                if self.busy {
                    ui.add(egui::Spinner::new().size(16.0).color(p.primary));
                }
                ui.label(theme.label(&self.status));
            });

            if let Some((done, total, file)) = &self.progress {
                if *total > 0 {
                    let ratio = (*done as f32 / *total as f32).clamp(0.0, 1.0);
                    ui.add_space(6.0);
                    theme::progress(ui, theme, ratio, 8.0);
                    ui.label(theme.label(crate::tf!("{:.1}%   {:.1} / {:.1} ГБ   {}", "{:.1}%   {:.1} / {:.1} GB   {}",
                        ratio * 100.0,
                        *done as f64 / 1e9,
                        *total as f64 / 1e9,
                        file
                    )));
                }
            }
            if !self.preview.is_empty() {
                ui.add_space(8.0);
                ui.label(theme.section(tr("Что внутри")));
                for line in &self.preview {
                    ui.label(
                        RichText::new(line)
                            .monospace()
                            .size(11.0)
                            .color(p.on_surface_variant),
                    );
                }
            }
            if !self.readme.is_empty() {
                ui.add_space(8.0);
                ui.label(theme.section("README"));
                Self::draw_readme(ui, theme, &self.readme);
            }
            if self.status == tr("Скачано") {
                action.replace(HubAction::Downloaded);
            }
        });
    }

    fn draw_on_device(
        &mut self,
        ui: &mut egui::Ui,
        theme: &Theme,
        local: &[ModelEntry],
        action: &mut Option<HubAction>,
    ) {
        let p = theme.pal;
        theme.card(p.surface_container_lowest, 12.0).show(ui, |ui| {
            ui.set_width(ui.available_width());
            ScrollArea::vertical()
                .id_salt("hub_on_device")
                .max_height(ui.available_height().max(220.0))
                .show(ui, |ui| {
                ui.set_width(ui.available_width());
                if local.is_empty() {
                    ui.label(theme.label(tr("Локальных моделей нет — скачай что-нибудь во вкладке Discover.")));
                }
                for m in local {
                    ui.horizontal(|ui| {
                        let mut line = m.path.clone();
                        if let Some(total) = m.total {
                            line.push_str(&format!("   · {}", crate::models::fmt_params(total)));
                        }
                        if let Some(act) = &m.active {
                            line.push_str(&format!(" ({act})"));
                        }
                        line.push_str(&crate::tf!("   · {:.0} МБ", "   · {:.0} MB", m.size_mb));
                        ui.label(RichText::new(line).size(13.5).color(p.on_surface));
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if theme::button(ui, theme, tr("Открыть в чате"), ButtonKind::Tonal, true).clicked() {
                                action.replace(HubAction::UseModel(m.path.clone()));
                            }
                        });
                    });
                }
            });
        });
    }
}
