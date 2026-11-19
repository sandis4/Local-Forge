//! Запуск Python-движка как дочернего процесса и обмен строками.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
#[cfg(windows)]
use std::os::windows::process::CommandExt;
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::{Duration, Instant};

pub enum Msg {
    Line(String),
}

pub struct Runner {
    child: Child,
    stdin: Option<ChildStdin>,
    rx: Receiver<Msg>,
    exit: Option<i32>,
}

impl Runner {
    pub fn spawn(python: &str, args: &[String], cwd: &Path) -> std::io::Result<Self> {
        let mut cmd = Command::new(python);
        #[cfg(windows)]
        {
            // CREATE_NO_WINDOW: иначе у Python-движка открывается консоль,
            // и это выглядит как вирус.
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        let mut child = cmd
            .args(args)
            .current_dir(cwd)
            // Без этого при перенаправленном выводе Python копит буфер
            // блоками, и дашборд молчит по несколько минут.
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin = child.stdin.take();
        let (tx, rx) = mpsc::channel();
        if let Some(out) = child.stdout.take() {
            pump(out, tx.clone());
        }
        if let Some(err) = child.stderr.take() {
            pump(err, tx);
        }
        Ok(Self { child, stdin, rx, exit: None })
    }

    /// Отправить команду в интерактивную консоль тренера (stop/save/pause/...).
    pub fn send(&mut self, cmd: &str) -> std::io::Result<()> {
        if let Some(si) = self.stdin.as_mut() {
            si.write_all(cmd.as_bytes())?;
            si.write_all(b"\n")?;
            si.flush()?;
        }
        Ok(())
    }

    pub fn drain(&self) -> Vec<String> {
        let mut out = Vec::new();
        loop {
            match self.rx.try_recv() {
                Ok(Msg::Line(l)) => out.push(l),
                Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
            }
        }
        out
    }

    pub fn poll_exit(&mut self) -> Option<i32> {
        if self.exit.is_some() {
            return self.exit;
        }
        match self.child.try_wait() {
            Ok(Some(status)) => {
                let code = status.code().unwrap_or(-1);
                self.exit = Some(code);
                Some(code)
            }
            _ => None,
        }
    }

    pub fn wait_exit(&mut self, timeout: Duration) -> Option<i32> {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Some(code) = self.poll_exit() {
                return Some(code);
            }
            thread::sleep(Duration::from_millis(100));
        }
        None
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
    }
}

fn pump<R: Read + Send + 'static>(r: R, tx: mpsc::Sender<Msg>) {
    thread::spawn(move || {
        let mut reader = BufReader::new(r);
        let mut buf: Vec<u8> = Vec::new();
        loop {
            buf.clear();
            match reader.read_until(b'\n', &mut buf) {
                Ok(0) => break,
                Ok(_) => {
                    let text = String::from_utf8_lossy(&buf);
                    let line = text.trim_end_matches(['\r', '\n']).to_string();
                    if tx.send(Msg::Line(line)).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });
}
