//! Material You: динамическая схема из одного seed-цвета.
//!
//! Тональные палитры строятся в CIELAB — упрощение официальной HCT
//! (там тон = L*, а оттенок/хрома берутся из CAM16). Роли M3 маппятся
//! на тона 0–100 так же, как в спецификации.

use eframe::egui::Color32;

use crate::theme::Palette;

pub const DEFAULT_SEED: [u8; 3] = [0x0E, 0x9F, 0x6E];

fn linearize(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

fn delinearize(c: f64) -> f64 {
    if c <= 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

fn rgb_to_lab(rgb: [u8; 3]) -> (f64, f64, f64) {
    let r = linearize(rgb[0] as f64 / 255.0);
    let g = linearize(rgb[1] as f64 / 255.0);
    let b = linearize(rgb[2] as f64 / 255.0);
    let x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b;
    fn f(t: f64) -> f64 {
        if t > 216.0 / 24389.0 {
            t.cbrt()
        } else {
            (24389.0 / 27.0 * t + 16.0) / 116.0
        }
    }
    let fx = f(x / 0.95047);
    let fy = f(y);
    let fz = f(z / 1.08883);
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

fn lab_to_rgb(l: f64, a: f64, b: f64) -> Color32 {
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    fn finv(t: f64) -> f64 {
        let t3 = t * t * t;
        if t3 > 216.0 / 24389.0 {
            t3
        } else {
            (116.0 * t - 16.0) / (24389.0 / 27.0)
        }
    }
    let x = finv(fx) * 0.95047;
    let y = finv(fy);
    let z = finv(fz) * 1.08883;
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let bl = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    let to = |v: f64| (delinearize(v.clamp(0.0, 1.0)) * 255.0).round().clamp(0.0, 255.0) as u8;
    Color32::from_rgb(to(r), to(g), to(bl))
}

#[derive(Clone, Copy)]
struct Lch {
    hue: f64,
    chroma: f64,
}

fn seed_lch(seed: [u8; 3]) -> Lch {
    let (_, a, b) = rgb_to_lab(seed);
    Lch {
        hue: b.atan2(a),
        chroma: (a * a + b * b).sqrt(),
    }
}

/// Тон палитры: L* фиксирован, хрома ограничена ролью.
fn tone(lch: Lch, tone: f64, chroma_cap: f64) -> Color32 {
    let c = lch.chroma.min(chroma_cap);
    lab_to_rgb(tone, c * lch.hue.cos(), c * lch.hue.sin())
}

/// Схема M3 (light/dark) из seed-цвета.
pub fn scheme(seed: [u8; 3], dark: bool) -> Palette {
    let primary = seed_lch(seed);
    // M3: secondary — тот же оттенок с малой хромой, tertiary — +60°.
    let secondary = Lch { hue: primary.hue, chroma: primary.chroma.min(16.0) };
    let tertiary = Lch { hue: primary.hue + 1.0472, chroma: primary.chroma.min(24.0) };
    let neutral = Lch { hue: primary.hue, chroma: 4.0 };
    let neutral_variant = Lch { hue: primary.hue, chroma: 8.0 };
    let error = seed_lch([0xB3, 0x26, 0x1E]);

    if dark {
        Palette {
            primary: tone(primary, 80.0, 48.0),
            on_primary: tone(primary, 20.0, 48.0),
            primary_container: tone(primary, 30.0, 48.0),
            on_primary_container: tone(primary, 90.0, 48.0),
            secondary: tone(secondary, 80.0, 16.0),
            on_secondary: tone(secondary, 20.0, 16.0),
            secondary_container: tone(secondary, 30.0, 16.0),
            on_secondary_container: tone(secondary, 90.0, 16.0),
            tertiary_container: tone(tertiary, 30.0, 24.0),
            on_tertiary_container: tone(tertiary, 90.0, 24.0),
            error: tone(error, 80.0, 60.0),
            on_error: tone(error, 20.0, 60.0),
            error_container: tone(error, 30.0, 60.0),
            on_error_container: tone(error, 90.0, 60.0),
            surface: tone(neutral, 6.0, 4.0),
            surface_container_lowest: tone(neutral, 4.0, 4.0),
            surface_container_low: tone(neutral, 10.0, 4.0),
            surface_container: tone(neutral, 12.0, 4.0),
            surface_container_high: tone(neutral, 17.0, 4.0),
            surface_container_highest: tone(neutral, 22.0, 4.0),
            on_surface: tone(neutral, 90.0, 4.0),
            on_surface_variant: tone(neutral_variant, 80.0, 8.0),
            outline: tone(neutral_variant, 60.0, 8.0),
            outline_variant: tone(neutral_variant, 30.0, 8.0),
            inverse_surface: tone(neutral, 90.0, 4.0),
            inverse_on_surface: tone(neutral, 20.0, 4.0),
        }
    } else {
        Palette {
            primary: tone(primary, 40.0, 48.0),
            on_primary: tone(primary, 100.0, 48.0),
            primary_container: tone(primary, 90.0, 48.0),
            on_primary_container: tone(primary, 10.0, 48.0),
            secondary: tone(secondary, 40.0, 16.0),
            on_secondary: tone(secondary, 100.0, 16.0),
            secondary_container: tone(secondary, 90.0, 16.0),
            on_secondary_container: tone(secondary, 10.0, 16.0),
            tertiary_container: tone(tertiary, 90.0, 24.0),
            on_tertiary_container: tone(tertiary, 10.0, 24.0),
            error: tone(error, 40.0, 60.0),
            on_error: tone(error, 100.0, 60.0),
            error_container: tone(error, 90.0, 60.0),
            on_error_container: tone(error, 10.0, 60.0),
            surface: tone(neutral, 99.0, 4.0),
            surface_container_lowest: tone(neutral, 100.0, 4.0),
            surface_container_low: tone(neutral, 96.0, 4.0),
            surface_container: tone(neutral, 94.0, 4.0),
            surface_container_high: tone(neutral, 92.0, 4.0),
            surface_container_highest: tone(neutral, 90.0, 4.0),
            on_surface: tone(neutral, 10.0, 4.0),
            on_surface_variant: tone(neutral_variant, 30.0, 8.0),
            outline: tone(neutral_variant, 50.0, 8.0),
            outline_variant: tone(neutral_variant, 80.0, 8.0),
            inverse_surface: tone(neutral, 20.0, 4.0),
            inverse_on_surface: tone(neutral, 95.0, 4.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheme_is_deterministic_and_contrasts() {
        let light = scheme(DEFAULT_SEED, false);
        let dark = scheme(DEFAULT_SEED, true);
        assert_ne!(light.primary, dark.primary);
        let lum = |c: Color32| c.r() as u32 + c.g() as u32 + c.b() as u32;
        // фон в светлой схеме светлее текста, в тёмной — наоборот
        assert!(lum(light.surface) > lum(light.on_surface));
        assert!(lum(dark.surface) < lum(dark.on_surface));
        // разные seed дают разные схемы
        assert_ne!(scheme([200, 40, 40], false).primary, scheme([40, 40, 200], false).primary);
    }
}
