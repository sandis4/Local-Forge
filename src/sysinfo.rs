//! Минимум системной информации без внешних крейтов: ядра и RAM.

pub fn cpu_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0)
}

#[cfg(windows)]
pub fn ram() -> (f64, f64) {
    #[repr(C)]
    struct MemStatus {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page: u64,
        avail_page: u64,
        total_virt: u64,
        avail_virt: u64,
        avail_ext: u64,
    }
    extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemStatus) -> i32;
    }
    let mut m: MemStatus = unsafe { std::mem::zeroed() };
    m.length = std::mem::size_of::<MemStatus>() as u32;
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) };
    if ok != 0 {
        (m.avail_phys as f64 / 1e9, m.total_phys as f64 / 1e9)
    } else {
        (0.0, 0.0)
    }
}

#[cfg(target_os = "linux")]
pub fn ram() -> (f64, f64) {
    let text = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    let mut total = 0.0;
    let mut avail = 0.0;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("MemTotal:") {
            total = v
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse::<f64>()
                .unwrap_or(0.0)
                / 1e6;
        } else if let Some(v) = line.strip_prefix("MemAvailable:") {
            avail = v
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse::<f64>()
                .unwrap_or(0.0)
                / 1e6;
        }
    }
    (avail, total)
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn ram() -> (f64, f64) {
    (0.0, 0.0)
}

/// Системный акцент Windows — ближайший аналог обоев для Material You.
#[cfg(windows)]
pub fn accent_color() -> Option<[u8; 3]> {
    extern "system" {
        fn DwmGetColorizationColor(color: *mut u32, opaque: *mut i32) -> i32;
    }
    let mut argb: u32 = 0;
    let mut opaque: i32 = 0;
    let ok = unsafe { DwmGetColorizationColor(&mut argb, &mut opaque) };
    if ok != 0 {
        let r = ((argb >> 16) & 0xFF) as u8;
        let g = ((argb >> 8) & 0xFF) as u8;
        let b = (argb & 0xFF) as u8;
        if r as u32 + g as u32 + b as u32 > 24 {
            return Some([r, g, b]);
        }
    }
    None
}

#[cfg(not(windows))]
pub fn accent_color() -> Option<[u8; 3]> {
    None
}
