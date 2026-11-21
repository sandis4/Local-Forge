//! Свой диспетчер папок в стиле Material 3.
//!
//! Показывает диски, быстрые места и подпапки, позволяет ввести путь руками.
//! Работает внутри окна приложения — без системного «Обзора папок».

use crate::i18n::tr;

fn home_dir() -> Option<PathBuf> {
    #[cfg(windows)]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(windows))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

fn default_root() -> PathBuf {
    #[cfg(windows)]
    {
        PathBuf::from("C:\\")
    }
    #[cfg(not(windows))]
    {
        PathBuf::from("/")
    }
}
use std::path::PathBuf;

use eframe::egui::{self, RichText};

use crate::theme::{self, ButtonKind, Theme};

pub enum PickerResult {
    Open,
    Picked(PathBuf),
    Cancelled,
}

pub struct FolderPicker {
    /// None — «Этот компьютер» со списком дисков.
    current: Option<PathBuf>,
    selected: Option<PathBuf>,
    path_input: String,
    error: String,
    entries: Vec<PathBuf>,
}

impl FolderPicker {
    pub fn new(start: Option<&str>) -> Self {
        let mut picker = Self {
            current: None,
            selected: None,
            path_input: String::new(),
            error: String::new(),
            entries: Vec::new(),
        };
        let start_dir = start
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .or_else(home_dir)
            .unwrap_or_else(default_root);
        if start_dir.is_dir() {
            picker.open(start_dir);
        } else {
            picker.refresh();
        }
        picker
    }

    fn refresh(&mut self) {
        self.selected = None;
        self.entries.clear();
        self.error.clear();
        match &self.current {
            None => {
                self.path_input = tr("Этот компьютер").into();
                #[cfg(windows)]
                {
                    for letter in b'A'..=b'Z' {
                        let root = PathBuf::from(format!("{}:\\", letter as char));
                        if root.is_dir() {
                            self.entries.push(root);
                        }
                    }
                }
                #[cfg(not(windows))]
                {
                    self.current = Some(PathBuf::from("/"));
                    self.refresh();
                    return;
                }
            }
            Some(dir) => {
                self.path_input = dir.display().to_string();
                match std::fs::read_dir(dir) {
                    Ok(rd) => {
                        for entry in rd.flatten() {
                            let path = entry.path();
                            if path.is_dir() {
                                self.entries.push(path);
                            }
                        }
                        self.entries.sort_by_key(|p| {
                            p.file_name()
                                .map(|s| s.to_string_lossy().to_lowercase())
                                .unwrap_or_default()
                        });
                    }
                    Err(err) => {
                        self.error = crate::tf!("Нет доступа: {err}", "No access: {err}");
                    }
                }
            }
        }
    }

    fn open(&mut self, dir: PathBuf) {
        self.current = Some(dir);
        self.refresh();
    }

    fn up(&mut self) {
        let Some(dir) = self.current.clone() else { return };
        match dir.parent() {
            Some(parent) if parent != dir => self.open(parent.to_path_buf()),
            _ => {
                self.current = None;
                self.refresh();
            }
        }
    }

    fn go_path(&mut self) {
        let text = self.path_input.trim().to_string();
        if text.is_empty() || text == tr("Этот компьютер") {
            self.current = None;
            self.refresh();
            return;
        }
        let path = PathBuf::from(&text);
        if path.is_dir() {
            self.open(path);
        } else {
            self.error = crate::tf!("Папка не найдена: {text}", "Folder not found: {text}");
        }
    }

    fn go_home(&mut self) {
        if let Some(home) = home_dir() {
            if home.is_dir() {
                self.open(home);
            }
        }
    }

    fn go_desktop(&mut self) {
        if let Some(home) = home_dir() {
            let path = home.join("Desktop");
            if path.is_dir() {
                self.open(path);
            }
        }
    }

    pub fn show(&mut self, ctx: &egui::Context, theme: &Theme) -> PickerResult {
        let p = theme.pal;
        let mut result = PickerResult::Open;

        // Затемнение и перехват кликов мимо диалога.
        egui::Area::new("folder_picker_scrim".into())
            .fixed_pos(egui::pos2(0.0, 0.0))
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                let screen = ctx.screen_rect();
                let (rect, resp) = ui.allocate_exact_size(
                    screen.size(),
                    egui::Sense::click_and_drag(),
                );
                ui.painter()
                    .rect_filled(rect, 0.0, egui::Color32::from_black_alpha(150));
                if resp.clicked() {
                    result = PickerResult::Cancelled;
                }
            });
        if ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            result = PickerResult::Cancelled;
        }

        egui::Window::new(tr("Выбор папки"))
            .id(egui::Id::new("folder_picker"))
            .order(egui::Order::Foreground)
            .title_bar(false)
            .resizable(false)
            .collapsible(false)
            .fixed_size(egui::vec2(640.0, 580.0))
            .anchor(egui::Align2::CENTER_CENTER, egui::vec2(0.0, 0.0))
            .frame(
                egui::Frame::none()
                    .fill(p.surface_container_high)
                    .rounding(egui::Rounding::same(28.0))
                    .inner_margin(egui::Margin::same(22.0))
                    .shadow(egui::epaint::Shadow {
                        offset: egui::vec2(0.0, 10.0),
                        blur: 28.0,
                        spread: 0.0,
                        color: egui::Color32::from_black_alpha(90),
                    }),
            )
            .show(ctx, |ui| {
                ui.set_width(596.0);
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(tr("Выбор папки"))
                            .font(theme::title_font(22.0))
                            .color(p.on_surface),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if close_button(ui, theme).clicked() {
                            result = PickerResult::Cancelled;
                        }
                    });
                });
                ui.add_space(2.0);
                ui.label(theme.label(tr("Папка станет проектом и рабочей папкой агента.")));
                ui.add_space(12.0);

                // Путь + переход
                ui.horizontal(|ui| {
                    if icon_button(ui, theme, Icon::Up).clicked() {
                        self.up();
                    }
                    let field = egui::Frame::none()
                        .fill(p.surface_container_highest)
                        .rounding(egui::Rounding::same(4.0))
                        .inner_margin(egui::Margin::symmetric(12.0, 9.0));
                    let resp = field
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.add(
                                egui::TextEdit::singleline(&mut self.path_input)
                                    .frame(false)
                                    .hint_text(tr("Введи путь и нажми Enter"))
                                    .desired_width(f32::INFINITY),
                            )
                        })
                        .inner;
                    if resp.lost_focus()
                        && ui.input(|i| i.key_pressed(egui::Key::Enter))
                    {
                        self.go_path();
                    }
                });
                ui.add_space(6.0);

                // Быстрые места
                ui.horizontal(|ui| {
                    if theme::button(ui, theme, tr("Этот компьютер"), ButtonKind::Text, true).clicked() {
                        self.current = None;
                        self.refresh();
                    }
                    if theme::button(ui, theme, tr("Домашняя"), ButtonKind::Text, true).clicked() {
                        self.go_home();
                    }
                    if theme::button(ui, theme, tr("Рабочий стол"), ButtonKind::Text, true).clicked() {
                        self.go_desktop();
                    }
                });
                ui.add_space(6.0);

                // Список папок
                let list_h = 300.0;
                egui::Frame::none()
                    .fill(p.surface_container_low)
                    .rounding(egui::Rounding::same(12.0))
                    .inner_margin(egui::Margin::symmetric(6.0, 6.0))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        egui::ScrollArea::vertical()
                            .max_height(list_h)
                            .auto_shrink([false, false])
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                if self.current.is_some() {
                                    if folder_row(ui, theme, None, false).double_clicked() {
                                        self.up();
                                    }
                                }
                                for path in self.entries.clone() {
                                    let selected = self.selected.as_ref() == Some(&path);
                                    let resp = folder_row(ui, theme, Some(&path), selected);
                                    if resp.clicked() {
                                        self.selected = Some(path.clone());
                                    }
                                    if resp.double_clicked() {
                                        self.open(path.clone());
                                        break;
                                    }
                                }
                                if self.entries.is_empty() && self.current.is_some() {
                                    ui.add_space(8.0);
                                    ui.label(theme.label(tr("Здесь нет подпапок.")));
                                }
                            });
                    });
                if !self.error.is_empty() {
                    ui.add_space(4.0);
                    ui.label(RichText::new(&self.error).size(12.0).color(p.error));
                }

                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    let target = self
                        .selected
                        .clone()
                        .or_else(|| self.current.clone());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if theme::button(
                            ui,
                            theme,
                            tr("Выбрать"),
                            ButtonKind::Filled,
                            target.is_some(),
                        )
                        .clicked()
                        {
                            if let Some(path) = target {
                                result = PickerResult::Picked(path);
                            }
                        }
                        if theme::button(ui, theme, tr("Отмена"), ButtonKind::Text, true).clicked() {
                            result = PickerResult::Cancelled;
                        }
                        ui.add(
                            egui::Label::new(
                                RichText::new(match &self.selected {
                                    Some(path) => path.display().to_string(),
                                    None => tr("Текущая папка").to_string(),
                                })
                                .size(12.0)
                                .color(p.on_surface_variant),
                            )
                            .truncate(),
                        );
                    });
                });
            });

        result
    }
}

enum Icon {
    Up,
}

fn icon_button(ui: &mut egui::Ui, theme: &Theme, icon: Icon) -> egui::Response {
    let p = theme.pal;
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(36.0, 36.0), egui::Sense::click());
    if resp.hovered() {
        ui.painter()
            .circle_filled(rect.center(), 18.0, p.surface_container_highest);
    }
    match icon {
        Icon::Up => theme::draw_icon(
            ui.painter(),
            rect.center(),
            theme::icons::ARROW_UP,
            18.0,
            p.on_surface_variant,
        ),
    }
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

fn close_button(ui: &mut egui::Ui, theme: &Theme) -> egui::Response {
    let p = theme.pal;
    let (rect, resp) =
        ui.allocate_exact_size(egui::vec2(36.0, 36.0), egui::Sense::click());
    if resp.hovered() {
        ui.painter()
            .circle_filled(rect.center(), 18.0, p.surface_container_highest);
    }
    theme::draw_icon(
        ui.painter(),
        rect.center(),
        theme::icons::CLOSE,
        16.0,
        p.on_surface_variant,
    );
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}

/// Строка папки; `path = None` — «наверх».
fn folder_row(
    ui: &mut egui::Ui,
    theme: &Theme,
    path: Option<&PathBuf>,
    selected: bool,
) -> egui::Response {
    let p = theme.pal;
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 34.0),
        egui::Sense::click(),
    );
    let fill = if selected {
        p.secondary_container
    } else if resp.hovered() {
        p.surface_container_high
    } else {
        egui::Color32::TRANSPARENT
    };
    let painter = ui.painter().clone();
    painter.rect_filled(rect, egui::Rounding::same(10.0), fill);
    let fg = if selected {
        p.on_secondary_container
    } else {
        p.on_surface
    };

    let icon_center = egui::pos2(rect.left() + 20.0, rect.center().y);
    theme::draw_icon(
        &painter,
        icon_center,
        if path.is_none() {
            theme::icons::ARROW_UP
        } else {
            theme::icons::FOLDER
        },
        16.0,
        fg,
    );

    let label = match path {
        None => tr("Наверх").to_string(),
        Some(path) => {
            let name = path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| path.display().to_string());
            // Для корня диска file_name() пустой — показываем «C:\».
            if name.is_empty() {
                path.display().to_string()
            } else {
                name
            }
        }
    };
    let max_chars = (((rect.width() - 64.0) / 7.0).max(6.0)) as usize;
    let label: String = if label.chars().count() > max_chars {
        format!("{}…", label.chars().take(max_chars.saturating_sub(1)).collect::<String>())
    } else {
        label
    };
    painter.text(
        egui::pos2(rect.left() + 40.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        theme::title_font(13.5),
        fg,
    );
    resp.on_hover_cursor(egui::CursorIcon::PointingHand)
}
