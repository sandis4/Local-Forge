//! Подсчёт параметров модели по самим весам: заголовок GGUF или
//! safetensors-хедеры. Имя файла («E2B») показывает только активные
//! параметры, поэтому ему доверять нельзя.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

fn read_u32<R: Read>(r: &mut R) -> Option<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).ok()?;
    Some(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> Option<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).ok()?;
    Some(u64::from_le_bytes(b))
}

fn read_str<R: Read>(r: &mut R) -> Option<String> {
    let n = read_u64(r)? as usize;
    if n > 16 * 1024 * 1024 {
        return None;
    }
    let mut b = vec![0u8; n];
    r.read_exact(&mut b).ok()?;
    Some(String::from_utf8_lossy(&b).into_owned())
}

/// Пропуск значения без чтения в память. `seek_relative` не сбрасывает
/// буфер BufReader — иначе сотни тысяч строк токенизатора в GGUF-метаданных
/// превращались в секунды.
fn skip_value(r: &mut BufReader<File>, kind: u32) -> Option<()> {
    match kind {
        0 | 1 | 7 => r.seek_relative(1).ok()?,
        2 | 3 => r.seek_relative(2).ok()?,
        4 | 5 | 6 => r.seek_relative(4).ok()?,
        10 | 11 | 12 => r.seek_relative(8).ok()?,
        8 => {
            let n = read_u64(r)? as i64;
            r.seek_relative(n).ok()?;
        }
        9 => {
            let elem = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_value(r, elem)?;
            }
        }
        _ => return None,
    }
    Some(())
}

/// Сумма размерностей всех тензоров из GGUF-заголовка.
pub fn gguf_params(path: &Path) -> Option<u64> {
    let file = File::open(path).ok()?;
    let mut r = BufReader::new(file);
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic).ok()?;
    if &magic != b"GGUF" {
        return None;
    }
    let _version = read_u32(&mut r)?;
    let n_tensors = read_u64(&mut r)?;
    let n_kv = read_u64(&mut r)?;
    for _ in 0..n_kv {
        read_str(&mut r)?;
        let kind = read_u32(&mut r)?;
        skip_value(&mut r, kind)?;
    }
    let mut total: u64 = 0;
    for _ in 0..n_tensors {
        read_str(&mut r)?;
        let nd = read_u32(&mut r)?;
        let mut count: u64 = 1;
        for _ in 0..nd {
            count = count.saturating_mul(read_u64(&mut r)?);
        }
        let _kind = read_u32(&mut r)?;
        let _offset = read_u64(&mut r)?;
        total = total.saturating_add(count);
    }
    Some(total)
}

/// Сумма параметров по всем shard'ам safetensors в папке.
pub fn safetensors_params(dir: &Path) -> Option<u64> {
    let mut total: u64 = 0;
    let mut found = false;
    for entry in std::fs::read_dir(dir).ok()? {
        let path = entry.ok()?.path();
        if !path
            .extension()
            .map(|e| e.eq_ignore_ascii_case("safetensors"))
            .unwrap_or(false)
        {
            continue;
        }
        let Ok(file) = File::open(&path) else { continue };
        let mut r = BufReader::new(file);
        let Some(size) = read_u64(&mut r) else { continue };
        if size == 0 || size > 64 * 1024 * 1024 {
            continue;
        }
        let mut buf = vec![0u8; size as usize];
        if r.read_exact(&mut buf).is_err() {
            continue;
        }
        let Ok(header) = serde_json::from_slice::<serde_json::Value>(&buf) else {
            continue;
        };
        let Some(obj) = header.as_object() else { continue };
        for (name, info) in obj {
            if name == "__metadata__" {
                continue;
            }
            let mut count: u64 = 1;
            if let Some(shape) = info.get("shape").and_then(|s| s.as_array()) {
                for d in shape {
                    count = count.saturating_mul(d.as_u64().unwrap_or(1));
                }
                total = total.saturating_add(count);
            }
        }
        found = true;
    }
    if found {
        Some(total)
    } else {
        None
    }
}

pub fn total_params(path: &Path) -> Option<u64> {
    if path.is_dir() {
        safetensors_params(path)
    } else {
        gguf_params(path)
    }
}

pub fn fmt_params(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.0}M", n as f64 / 1e6)
    } else {
        format!("{:.0}K", n as f64 / 1e3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_sizes() {
        assert_eq!(fmt_params(4_628_569_635), "4.6B");
        assert_eq!(fmt_params(135_000_000), "135M");
        assert_eq!(fmt_params(50_000), "50K");
    }
}

#[cfg(test)]
mod real_tests {
    use super::*;

    #[test]
    #[ignore]
    fn real_file_counts() {
        let ws = std::path::Path::new("..");
        let q4 = total_params(&ws.join("models_gguf/Forge-1-E2B-Q4_K_M.gguf")).unwrap();
        eprintln!("E2B Q4: {} ({})", fmt_params(q4), q4);
        assert!(q4 > 4_000_000_000 && q4 < 6_000_000_000);
        let qwen = total_params(&ws.join("models_gguf/Forge1-Qwen3.5-9B-Q4_K_M.gguf")).unwrap();
        eprintln!("Qwen 9B: {} ({})", fmt_params(qwen), qwen);
        assert!(qwen > 8_000_000_000 && qwen < 11_000_000_000);
        let gemma = total_params(&ws.join("models/gemma_base")).unwrap();
        eprintln!("gemma_base: {} ({})", fmt_params(gemma), gemma);
        assert!(gemma > 4_500_000_000);
    }
}
