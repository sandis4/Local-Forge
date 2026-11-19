//! Material 3 (Material You) — палитра, типографика и базовые компоненты.
//!
//! Цвета — официальная базовая схема Material 3 (light/dark): tonal-поверхности,
//! primary/secondary-контейнеры, outline. Формы — пилюли у кнопок, 16–28 у карточек.

use eframe::egui::{
    self, Color32, FontFamily, FontId, RichText, Rounding, Stroke, TextStyle, Ui,
};

#[allow(dead_code)] // полная схема M3: часть ролей нужна в других состояниях
#[derive(Clone, Copy)]
pub struct Palette {
    pub primary: Color32,
    pub on_primary: Color32,
    pub primary_container: Color32,
    pub on_primary_container: Color32,
    pub secondary: Color32,
    pub on_secondary: Color32,
    pub secondary_container: Color32,
    pub on_secondary_container: Color32,
    pub tertiary_container: Color32,
    pub on_tertiary_container: Color32,
    pub error: Color32,
    pub on_error: Color32,
    pub error_container: Color32,
    pub on_error_container: Color32,
    pub surface: Color32,
    pub surface_container_lowest: Color32,
    pub surface_container_low: Color32,
    pub surface_container: Color32,
    pub surface_container_high: Color32,
    pub surface_container_highest: Color32,
    pub on_surface: Color32,
    pub on_surface_variant: Color32,
    pub outline: Color32,
    pub outline_variant: Color32,
    pub inverse_surface: Color32,
    pub inverse_on_surface: Color32,
}

/// Material Symbols Rounded: коды глифов для интерфейсных иконок.
pub mod icons {
    pub const CHAT: char = '\u{e0cb}';
    pub const DASHBOARD: char = '\u{e871}';
    pub const TUNE: char = '\u{e429}';
    pub const SETTINGS: char = '\u{e8b8}';
    pub const ADD: char = '\u{e145}';
    pub const FOLDER: char = '\u{e2c7}';
    pub const CLOSE: char = '\u{e5cd}';
    pub const SEARCH: char = '\u{ef7a}';
    pub const EXPAND_MORE: char = '\u{e5cf}';
    pub const DOWNLOAD: char = '\u{f090}';
    pub const ARROW_BACK: char = '\u{e5c4}';
    pub const ARROW_UP: char = '\u{e5d8}';
    pub const MINIMIZE: char = '\u{e15b}';
    pub const MAXIMIZE: char = '\u{e3c6}';
}

pub fn icon_font(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name("m3-icons".into()))
}

pub fn draw_icon(
    painter: &egui::Painter,
    center: egui::Pos2,
    code: char,
    size: f32,
    color: Color32,
) {
    painter.text(
        center,
        egui::Align2::CENTER_CENTER,
        code,
        icon_font(size),
        color,
    );
}

/// Семейство жирного шрифта (Roboto Bold).
pub fn bold_family() -> FontFamily {
    FontFamily::Name("m3-bold".into())
}

/// Roboto Medium — M3 использует его для title/label.
pub fn medium_family() -> FontFamily {
    FontFamily::Name("m3-medium".into())
}

pub fn title_font(size: f32) -> FontId {
    FontId::new(size, medium_family())
}

#[derive(Clone, Copy)]
pub struct Theme {
    pub dark: bool,
    pub pal: Palette,
}

pub fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let f = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).clamp(0.0, 255.0) as u8;
    Color32::from_rgba_premultiplied(
        f(a.r(), b.r()),
        f(a.g(), b.g()),
        f(a.b(), b.b()),
        f(a.a(), b.a()),
    )
}

pub fn lerp_palette(a: &Palette, b: &Palette, t: f32) -> Palette {
    macro_rules! c {
        ($f:ident) => {
            lerp_color(a.$f, b.$f, t)
        };
    }
    Palette {
        primary: c!(primary),
        on_primary: c!(on_primary),
        primary_container: c!(primary_container),
        on_primary_container: c!(on_primary_container),
        secondary: c!(secondary),
        on_secondary: c!(on_secondary),
        secondary_container: c!(secondary_container),
        on_secondary_container: c!(on_secondary_container),
        tertiary_container: c!(tertiary_container),
        on_tertiary_container: c!(on_tertiary_container),
        error: c!(error),
        on_error: c!(on_error),
        error_container: c!(error_container),
        on_error_container: c!(on_error_container),
        surface: c!(surface),
        surface_container_lowest: c!(surface_container_lowest),
        surface_container_low: c!(surface_container_low),
        surface_container: c!(surface_container),
        surface_container_high: c!(surface_container_high),
        surface_container_highest: c!(surface_container_highest),
        on_surface: c!(on_surface),
        on_surface_variant: c!(on_surface_variant),
        outline: c!(outline),
        outline_variant: c!(outline_variant),
        inverse_surface: c!(inverse_surface),
        inverse_on_surface: c!(inverse_on_surface),
    }
}

impl Theme {
    pub fn new(dark: bool) -> Self {
        Self::from_seed(crate::color::DEFAULT_SEED, dark)
    }

    /// Material You: схема целиком выводится из одного seed-цвета.
    pub fn from_seed(seed: [u8; 3], dark: bool) -> Self {
        Self {
            dark,
            pal: crate::color::scheme(seed, dark),
        }
    }

    /// Roboto — шрифт Material 3. Встроен в exe, чтобы вид не зависел от системы.
    pub fn install_fonts(ctx: &egui::Context) {
        use egui::{FontData, FontDefinitions, FontFamily};

        let mut fonts = FontDefinitions::default();
        fonts.font_data.insert(
            "m3-sans".to_owned(),
            FontData::from_static(include_bytes!("../assets/fonts/Roboto-Regular.ttf")),
        );
        fonts.font_data.insert(
            "m3-sans-medium".to_owned(),
            FontData::from_static(include_bytes!("../assets/fonts/Roboto-Medium.ttf")),
        );
        fonts.font_data.insert(
            "m3-sans-bold".to_owned(),
            FontData::from_static(include_bytes!("../assets/fonts/Roboto-Bold.ttf")),
        );
        fonts.font_data.insert(
            "m3-mono".to_owned(),
            FontData::from_static(include_bytes!("../assets/fonts/RobotoMono-Regular.ttf")),
        );
        fonts.font_data.insert(
            "m3-icons".to_owned(),
            FontData::from_static(include_bytes!("../assets/fonts/MaterialSymbols-subset.ttf")),
        );

        if let Some(list) = fonts.families.get_mut(&FontFamily::Proportional) {
            list.insert(0, "m3-sans".to_owned());
        }
        if let Some(list) = fonts.families.get_mut(&FontFamily::Monospace) {
            list.insert(0, "m3-mono".to_owned());
        }
        fonts.families.insert(
            FontFamily::Name("m3-bold".into()),
            vec!["m3-sans-bold".to_owned(), "m3-sans".to_owned()],
        );
        fonts.families.insert(
            FontFamily::Name("m3-medium".into()),
            vec!["m3-sans-medium".to_owned(), "m3-sans".to_owned()],
        );
        fonts.families.insert(
            FontFamily::Name("m3-icons".into()),
            vec!["m3-icons".to_owned()],
        );
        ctx.set_fonts(fonts);
    }

    pub fn apply(&self, ctx: &egui::Context) {
        let p = &self.pal;
        let mut style = (*ctx.style()).clone();
        let v = &mut style.visuals;
        v.dark_mode = self.dark;
        v.override_text_color = Some(p.on_surface);
        v.panel_fill = p.surface;
        v.window_fill = p.surface;
        v.extreme_bg_color = p.surface_container_highest;
        v.faint_bg_color = p.surface_container;
        v.selection.bg_fill = p.primary_container;
        v.selection.stroke = Stroke::new(1.0_f32, p.on_primary_container);
        v.hyperlink_color = p.primary;
        v.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, p.on_surface_variant);
        v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, p.outline_variant);
        v.widgets.noninteractive.rounding = Rounding::same(12.0);
        v.widgets.inactive.fg_stroke = Stroke::new(1.0_f32, p.on_surface);
        v.widgets.inactive.bg_fill = p.surface_container_high;
        v.widgets.inactive.weak_bg_fill = p.surface_container_high;
        v.widgets.inactive.rounding = Rounding::same(4.0);
        v.widgets.inactive.expansion = 0.0;
        v.widgets.hovered.bg_fill = p.surface_container_highest;
        v.widgets.hovered.weak_bg_fill = p.surface_container_highest;
        v.widgets.hovered.rounding = Rounding::same(4.0);
        v.widgets.hovered.expansion = 0.0;
        v.widgets.active.bg_fill = p.primary_container;
        v.widgets.active.weak_bg_fill = p.primary_container;
        v.widgets.active.rounding = Rounding::same(4.0);
        v.widgets.active.expansion = 0.0;
        v.widgets.open.bg_fill = p.surface_container_highest;
        v.widgets.open.rounding = Rounding::same(4.0);

        // Сетка M3: 4dp-шаг, компактные контролы 40dp.
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.button_padding = egui::vec2(16.0, 8.0);
        style.spacing.interact_size.y = 40.0;
        style.spacing.menu_margin = egui::Margin::same(6.0);
        style.spacing.scroll.bar_width = 8.0;
        style.spacing.scroll.floating = true;
        style.spacing.scroll.floating_width = 5.0;
        // Шкала M3: labelLarge 14 medium, bodyMedium 14, titleLarge 22,
        // headlineSmall 24, displaySmall 36.
        style.text_styles = [
            (
                TextStyle::Heading,
                FontId::new(24.0, medium_family()),
            ),
            (TextStyle::Body, FontId::new(14.5, FontFamily::Proportional)),
            (TextStyle::Button, FontId::new(14.0, medium_family())),
            (TextStyle::Small, FontId::new(12.0, FontFamily::Proportional)),
            (TextStyle::Monospace, FontId::new(12.5, FontFamily::Monospace)),
        ]
        .into();
        ctx.set_style(style);
    }

    /// Плоская тональная секция: без «пластиковой» обводки, радиус 8.
    pub fn card(&self, fill: Color32, _radius: f32) -> egui::Frame {
        egui::Frame::none()
            .fill(fill)
            .rounding(Rounding::same(8.0))
            .inner_margin(egui::Margin::symmetric(20.0, 16.0))
    }

    /// Надзаголовок: моно, капс, акцентный цвет.
    pub fn overline(&self, text: impl Into<String>) -> RichText {
        RichText::new(text.into().to_uppercase())
            .font(FontId::new(10.5, FontFamily::Monospace))
            .color(self.pal.primary)
    }

    /// Крупный заголовок-плакат — главный типографический ход.
    pub fn display(&self, text: impl Into<String>, size: f32) -> RichText {
        RichText::new(text)
            .font(FontId::new(size, FontFamily::Proportional))
            .color(self.pal.on_surface)
    }

    pub fn mono(&self, text: impl Into<String>, size: f32) -> RichText {
        RichText::new(text)
            .font(FontId::new(size, FontFamily::Monospace))
            .color(self.pal.on_surface_variant)
    }

    /// Волосяная линия вместо рамки карточки.
    pub fn hairline(&self, ui: &mut Ui) {
        let (rect, _) = ui.allocate_exact_size(
            egui::vec2(ui.available_width(), 1.0),
            egui::Sense::hover(),
        );
        ui.painter()
            .line_segment([rect.left_bottom(), rect.right_bottom()], Stroke::new(1.0_f32, self.pal.outline_variant));
    }

    pub fn label(&self, text: impl Into<String>) -> RichText {
        RichText::new(text).color(self.pal.on_surface_variant).size(13.0)
    }

    /// Заголовок секции внутри карточки: моно-капс, тише основного текста.
    pub fn section(&self, text: impl Into<String>) -> RichText {
        RichText::new(text.into().to_uppercase())
            .font(FontId::new(10.5, FontFamily::Monospace))
            .color(self.pal.on_surface_variant)
    }

    pub fn value(&self, text: impl Into<String>) -> RichText {
        RichText::new(text)
            .font(FontId::new(22.0, bold_family()))
            .color(self.pal.on_surface)
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum ButtonKind {
    Filled,
    Tonal,
    Outlined,
    Text,
    Danger,
}

/// Кнопка в стиле M3: пилюля, три уровня тональности.
pub fn button(ui: &mut Ui, theme: &Theme, text: &str, kind: ButtonKind, enabled: bool) -> egui::Response {
    let p = &theme.pal;
    let (fill, fg, stroke) = match kind {
        ButtonKind::Filled => (p.primary, p.on_primary, Stroke::NONE),
        ButtonKind::Tonal => (p.secondary_container, p.on_secondary_container, Stroke::NONE),
        ButtonKind::Outlined => (Color32::TRANSPARENT, p.primary, Stroke::new(1.0_f32, p.outline)),
        ButtonKind::Text => (Color32::TRANSPARENT, p.primary, Stroke::NONE),
        ButtonKind::Danger => (p.error, p.on_error, Stroke::NONE),
    };
    let fg = if enabled { fg } else { p.on_surface_variant };
    // У текстовой/контурной кнопки в disabled-состоянии фона быть не должно.
    let fill = if enabled {
        fill
    } else {
        match kind {
            ButtonKind::Filled | ButtonKind::Tonal | ButtonKind::Danger => {
                p.surface_container_high
            }
            ButtonKind::Outlined | ButtonKind::Text => Color32::TRANSPARENT,
        }
    };
    let stroke = if enabled {
        stroke
    } else if matches!(kind, ButtonKind::Outlined) {
        Stroke::new(1.0_f32, p.outline_variant)
    } else {
        Stroke::NONE
    };
    let mut button = egui::Button::new(
        RichText::new(text)
            .color(fg)
            .font(FontId::new(14.0, medium_family())),
    )
    .fill(fill)
    .stroke(stroke)
    .rounding(Rounding::same(match kind {
        ButtonKind::Filled | ButtonKind::Danger | ButtonKind::Tonal => 10.0,
        ButtonKind::Outlined => 4.0,
        ButtonKind::Text => 0.0,
    }))
    .min_size(egui::vec2(0.0, 40.0));
    if !enabled {
        button = button.sense(egui::Sense::hover());
    }
    ui.add(button)
}

/// M3 segmented button: один контейнер, выбранный сегмент — tonal.
pub fn segmented(ui: &mut Ui, theme: &Theme, options: &[&str], selected: &str) -> Option<String> {
    let p = &theme.pal;
    let mut clicked = None;
    egui::Frame::none()
        .fill(p.surface_container_high)
        .rounding(Rounding::same(6.0))
        .inner_margin(egui::Margin::same(3.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                for opt in options {
                    let is_sel = *opt == selected;
                    let (fill, fg) = if is_sel {
                        (p.surface_container_lowest, p.on_surface)
                    } else {
                        (Color32::TRANSPARENT, p.on_surface_variant)
                    };
                    let resp = ui.add(
                        egui::Button::new(
                            RichText::new(*opt)
                                .color(fg)
                                .font(FontId::new(14.0, medium_family())),
                        )
                        .fill(fill)
                        .stroke(Stroke::NONE)
                        .rounding(Rounding::same(4.0))
                        .min_size(egui::vec2(0.0, 32.0)),
                    );
                    if resp.clicked() {
                        clicked = Some((*opt).to_string());
                    }
                }
            });
        });
    clicked
}

/// M3 switch: дорожка 40×22, бегунок 16; клик по подписи тоже переключает.
pub fn switch(ui: &mut Ui, theme: &Theme, on: &mut bool, label: &str) -> bool {
    let p = &theme.pal;
    let font = FontId::new(13.0, medium_family());
    let text_w = label.chars().count() as f32 * 7.4;
    let (rect, resp) = ui.allocate_exact_size(
        egui::vec2(40.0 + 8.0 + text_w, 28.0),
        egui::Sense::click(),
    );
    let track = egui::Rect::from_center_size(
        egui::pos2(rect.left() + 20.0, rect.center().y),
        egui::vec2(40.0, 22.0),
    );
    let fill = if *on {
        if resp.hovered() {
            lerp_color(p.primary, p.on_primary, 0.08)
        } else {
            p.primary
        }
    } else if resp.hovered() {
        p.surface_container_highest
    } else {
        p.surface_container_high
    };
    let painter = ui.painter();
    painter.rect_filled(track, Rounding::same(11.0), fill);
    if !*on {
        painter.rect_stroke(
            track,
            Rounding::same(11.0),
            Stroke::new(1.0_f32, p.outline),
        );
    }
    let thumb_x = if *on {
        track.right() - 11.0
    } else {
        track.left() + 11.0
    };
    painter.circle_filled(
        egui::pos2(thumb_x, track.center().y),
        8.0,
        if *on { p.on_primary } else { p.outline },
    );
    painter.text(
        egui::pos2(rect.left() + 48.0, rect.center().y),
        egui::Align2::LEFT_CENTER,
        label,
        font,
        if *on { p.on_surface } else { p.on_surface_variant },
    );
    let clicked = resp.on_hover_cursor(egui::CursorIcon::PointingHand).clicked();
    if clicked {
        *on = !*on;
    }
    clicked
}

/// Чип с иконкой слева (например, счётчик скачиваний).
pub fn icon_chip(ui: &mut Ui, theme: &Theme, code: char, text: &str) {
    egui::Frame::none()
        .fill(theme.pal.surface_container_high)
        .rounding(Rounding::same(4.0))
        .inner_margin(egui::Margin::symmetric(6.0, 2.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 5.0;
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(11.0, 11.0), egui::Sense::hover());
                draw_icon(
                    ui.painter(),
                    rect.center(),
                    code,
                    11.0,
                    theme.pal.on_surface_variant,
                );
                ui.add(
                    egui::Label::new(
                        RichText::new(text)
                            .size(11.5)
                            .color(theme.pal.on_surface_variant),
                    )
                    .wrap_mode(egui::TextWrapMode::Extend),
                );
            });
        });
}

/// Цветной тег в духе Hugging Face: тонированный фон + подпись.
pub fn tag_chip(ui: &mut Ui, theme: &Theme, text: &str, color: Color32) {
    let fill = lerp_color(theme.pal.surface_container_high, color, 0.20);
    egui::Frame::none()
        .fill(fill)
        .rounding(Rounding::same(4.0))
        .inner_margin(egui::Margin::symmetric(4.0, 1.0))
        .show(ui, |ui| {
            ui.add(
                egui::Label::new(
                    RichText::new(text)
                        .size(10.0)
                        .color(lerp_color(color, theme.pal.on_surface, 0.18)),
                )
                .wrap_mode(egui::TextWrapMode::Extend),
            );
        });
}

/// M3 линейный индикатор прогресса с закруглёнными концами.
pub fn progress(ui: &mut Ui, theme: &Theme, ratio: f32, height: f32) {
    let p = &theme.pal;
    let width = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(egui::vec2(width, height), egui::Sense::hover());
    let radius = height / 2.0;
    let painter = ui.painter();
    painter.rect_filled(rect, Rounding::same(radius), p.surface_container_highest);
    let ratio = ratio.clamp(0.0, 1.0);
    if ratio > 0.0 {
        let filled = egui::Rect::from_min_size(
            rect.min,
            egui::vec2((rect.width() * ratio).max(height), rect.height()),
        );
        painter.rect_filled(filled, Rounding::same(radius), p.primary);
    }
}

/// Маленький чип (M3 assist chip): радиус 8, контур outline_variant.
pub fn chip(ui: &mut Ui, theme: &Theme, text: &str) {
    egui::Frame::none()
        .fill(theme.pal.surface_container_high)
        .rounding(Rounding::same(4.0))
        .inner_margin(egui::Margin::symmetric(6.0, 2.0))
        .show(ui, |ui| {
            ui.add(
                egui::Label::new(
                    RichText::new(text).size(11.0).color(theme.pal.on_surface_variant),
                )
                .wrap_mode(egui::TextWrapMode::Extend),
            );
        });
}
