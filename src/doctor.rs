//! Проверка окружения: железо + автоплан. Считает тот же Python-код,
//! что и тренер (hardware.py + memory_plan.py), поэтому цифры не расходятся.

use std::path::Path;
use std::process::Command;
#[cfg(windows)]
use std::os::windows::process::CommandExt;

pub const SCRIPT: &str = r#"
import sys
import hardware, memory_plan

hw = hardware.detect_hardware()
print(hardware.describe_hardware(hw))

model = sys.argv[1] if len(sys.argv) > 1 else ""
try:
    plan = memory_plan.plan_for_model(model, hw, user={})
    print(memory_plan.describe_plan(plan))
    for w in plan.get("warnings", []):
        print("  [!]", w)
except Exception as e:
    print("Не удалось построить автоплан:", e)
"#;

pub fn run(python: &str, engine: &Path, model: &Path) -> std::io::Result<i32> {
    println!("python: {python}");
    println!("{}", crate::tf!("движок: {}", "engine: {}", engine.display()));
    println!("{}", crate::tf!("модель: {}", "model: {}", model.display()));
    println!();
    let mut cmd = Command::new(python);
    #[cfg(windows)]
    {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let status = cmd
        .arg("-c")
        .arg(SCRIPT)
        .arg(model)
        .current_dir(engine)
        .env("PYTHONIOENCODING", "utf-8")
        .status()?;
    Ok(status.code().unwrap_or(-1))
}
