//! Дашборд и два режима работы: TUI для интерактивной консоли
//! и plain-режим для перенаправленного вывода/скриптов.

use crate::i18n::tr;
use std::collections::VecDeque;
use std::io::{self, IsTerminal};
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, Paragraph};
use ratatui::{Frame, Terminal};

use crate::parse::{human_duration, Stats};
use crate::runner::Runner;

const LOG_LIMIT: usize = 500;

pub struct App {
    pub stats: Stats,
    pub log: VecDeque<String>,
    pub paused: bool,
    pub quitting: bool,
    pub exit_code: Option<i32>,
}

impl App {
    pub fn new() -> Self {
        Self {
            stats: Stats::default(),
            log: VecDeque::new(),
            paused: false,
            quitting: false,
            exit_code: None,
        }
    }

    pub fn push_line(&mut self, line: String) {
        let was_step = self.stats.feed(&line);
        if was_step && self.stats.started.is_none() {
            // Отсчёт ETA — от первого шага, а не от старта процесса:
            // загрузка модели не должна попадать в прогноз.
            self.stats.begin();
        }
        if !was_step {
            self.push_capped(line);
        }
    }

    pub fn push_log(&mut self, line: impl Into<String>) {
        self.push_capped(line.into());
    }

    /// Журнал ограничен LOG_LIMIT строк: старые вытесняются.
    fn push_capped(&mut self, line: String) {
        self.log.push_back(line);
        while self.log.len() > LOG_LIMIT {
            self.log.pop_front();
        }
    }
}

fn draw_header(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default()
        .borders(Borders::ALL)
        .title(Span::styled(" Forge Train ", Style::default().add_modifier(Modifier::BOLD)));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1), Constraint::Length(1), Constraint::Length(1)])
        .split(inner);

    let s = &app.stats;
    let model = s.model.clone().unwrap_or_else(|| "—".into());
    let profile = s.profile.clone().unwrap_or_else(|| "—".into());
    let padding = s.padding.clone().unwrap_or_else(|| tr("динамический").into());

    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(tr("модель: "), Style::default().fg(Color::DarkGray)),
            Span::raw(model),
            Span::styled(tr("   профиль: "), Style::default().fg(Color::DarkGray)),
            Span::raw(profile),
            Span::styled("   padding: ", Style::default().fg(Color::DarkGray)),
            Span::raw(padding),
        ])),
        rows[0],
    );

    let total = s.total_steps.unwrap_or(0);
    let label = if total > 0 {
        crate::tf!("шаг {}/{}  {:.1}%", "step {}/{}  {:.1}%", s.step, total, s.progress() * 100.0)
    } else {
        crate::tf!("шаг {}", "step {}", s.step)
    };
    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(Color::Green))
        .ratio(s.progress())
        .label(label);
    f.render_widget(gauge, rows[1]);

    let loss = s.loss.map(|v| format!("{v:.4}")).unwrap_or_else(|| "—".into());
    let lr = s.lr.map(|v| format!("{v:.2e}")).unwrap_or_else(|| "—".into());
    let ram = s.ram_mb.map(|v| crate::tf!("{v} МБ", "{v} MB")).unwrap_or_else(|| "—".into());
    let threads = s.threads.map(|v| v.to_string()).unwrap_or_else(|| "—".into());
    let epoch = match (s.epoch, s.epochs) {
        (Some(e), Some(n)) => format!("{e}/{n}"),
        _ => "—".into(),
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled("loss ", Style::default().fg(Color::DarkGray)),
            Span::styled(loss, Style::default().fg(Color::Yellow)),
            Span::styled("   lr ", Style::default().fg(Color::DarkGray)),
            Span::raw(lr),
            Span::styled("   RAM ", Style::default().fg(Color::DarkGray)),
            Span::raw(ram),
            Span::styled(tr("   потоков "), Style::default().fg(Color::DarkGray)),
            Span::raw(threads),
            Span::styled(tr("   эпоха "), Style::default().fg(Color::DarkGray)),
            Span::raw(epoch),
        ])),
        rows[2],
    );

    let status = if s.finished {
        tr("готово").to_string()
    } else if app.quitting {
        tr("останавливаю и сохраняю…").to_string()
    } else if app.paused {
        tr("ПАУЗА").to_string()
    } else {
        tr("идёт обучение").to_string()
    };
    let eta = s.eta().map(human_duration).unwrap_or_else(|| "—".into());
    let status_style = if app.paused {
        Style::default().fg(Color::Yellow)
    } else if app.quitting {
        Style::default().fg(Color::Red)
    } else {
        Style::default().fg(Color::Green)
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(status, status_style),
            Span::styled(tr("   осталось "), Style::default().fg(Color::DarkGray)),
            Span::raw(eta),
            Span::styled(tr("   сохранений "), Style::default().fg(Color::DarkGray)),
            Span::raw(s.saves.to_string()),
        ])),
        rows[3],
    );
}

fn draw_log(f: &mut Frame, app: &App, area: Rect) {
    let block = Block::default().borders(Borders::ALL).title(tr(" журнал "));
    let inner = block.inner(area);
    f.render_widget(block, area);

    let height = inner.height as usize;
    let skip = app.log.len().saturating_sub(height);
    let lines: Vec<Line> = app
        .log
        .iter()
        .skip(skip)
        .map(|l| Line::from(l.clone()))
        .collect();
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_footer(f: &mut Frame, app: &App, area: Rect) {
    let hint = if app.stats.error.is_some() {
        tr("ОШИБКА — смотри журнал выше")
    } else {
        tr("1 game  2 work  3 max  4 turbo   p пауза/продолжить   s сохранить   t статус   x стоп   q выход")
    };
    let style = if app.stats.error.is_some() {
        Style::default().fg(Color::Red)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    f.render_widget(Paragraph::new(Line::from(Span::styled(hint, style))), area);
}

pub fn draw(f: &mut Frame, app: &App) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(6), Constraint::Min(4), Constraint::Length(1)])
        .split(f.area());
    draw_header(f, app, rows[0]);
    draw_log(f, app, rows[1]);
    draw_footer(f, app, rows[2]);
}

pub fn run_tui(mut runner: Runner, mut app: App) -> io::Result<i32> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, crossterm::cursor::Hide)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = event_loop(&mut terminal, &mut runner, &mut app);

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen, crossterm::cursor::Show)?;
    let _ = terminal.show_cursor();
    result?;
    Ok(app.exit_code.unwrap_or(-1))
}

fn event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    runner: &mut Runner,
    app: &mut App,
) -> io::Result<()> {
    let mut stop_deadline: Option<Instant> = None;
    loop {
        for line in runner.drain() {
            app.push_line(line);
        }
        if let Some(code) = runner.poll_exit() {
            app.exit_code = Some(code);
            break;
        }
        if app.quitting {
            let deadline =
                *stop_deadline.get_or_insert_with(|| Instant::now() + Duration::from_secs(120));
            if Instant::now() > deadline {
                runner.kill();
                app.exit_code = runner.wait_exit(Duration::from_secs(5));
                break;
            }
        }

        terminal.draw(|f| draw(f, app))?;

        if event::poll(Duration::from_millis(120))? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    handle_key(key, runner, app);
                }
            }
        }
    }
    Ok(())
}

fn handle_key(key: KeyEvent, runner: &mut Runner, app: &mut App) {
    let cmd = match key.code {
        KeyCode::Char('1') => Some("game"),
        KeyCode::Char('2') => Some("work"),
        KeyCode::Char('3') => Some("max"),
        KeyCode::Char('4') => Some("turbo"),
        KeyCode::Char('p') => {
            app.paused = !app.paused;
            Some(if app.paused { "pause" } else { "resume" })
        }
        KeyCode::Char('s') => Some("save"),
        KeyCode::Char('t') => Some("status"),
        KeyCode::Char('x') | KeyCode::Char('q') | KeyCode::Esc => {
            app.quitting = true;
            Some("stop")
        }
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            app.quitting = true;
            Some("stop")
        }
        _ => None,
    };
    if let Some(c) = cmd {
        let _ = runner.send(c);
        app.push_log(format!("> {c}"));
    }
}

/// Перенаправленный вывод или флаг --plain: просто транслируем строки,
/// а набранное в консоли отправляем тренеру (как при обычном запуске python).
pub fn run_plain(mut runner: Runner, mut app: App) -> io::Result<i32> {
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lines() {
            match line {
                Ok(l) => {
                    if tx.send(l).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    loop {
        while let Ok(cmd) = rx.try_recv() {
            let _ = runner.send(&cmd);
        }
        for line in runner.drain() {
            println!("{line}");
            app.push_line(line);
        }
        if let Some(code) = runner.poll_exit() {
            app.exit_code = Some(code);
            return Ok(code);
        }
        thread::sleep(Duration::from_millis(50));
    }
}

pub fn stdout_is_tty() -> bool {
    io::stdout().is_terminal()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;

    fn render(app: &App, w: u16, h: u16) -> String {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).unwrap();
        terminal.draw(|f| draw(f, app)).unwrap();
        let buffer = terminal.backend().buffer();
        let mut out = String::new();
        for y in 0..buffer.area.height {
            for x in 0..buffer.area.width {
                out.push_str(buffer[(x, y)].symbol());
            }
            out.push('\n');
        }
        out
    }

    #[test]
    fn renders_status_fields() {
        let mut app = App::new();
        app.push_line("[Шаг 42] Loss: 0.8342 | LR: 1.23e-05 | RAM процесса: 2345MB | Потоков: 8 | Профиль: work".to_string());
        app.push_log("[Шедулер] -> шагов ~100, warmup 5.");
        app.stats.total_steps = Some(100);
        app.stats.model = Some("models/gemma_base".into());
        let screen = render(&app, 110, 18);
        assert!(screen.contains("Forge Train"), "{screen}");
        assert!(screen.contains("0.8342"), "{screen}");
        assert!(screen.contains("42/100"), "{screen}");
        assert!(screen.contains("2345"), "{screen}");
        assert!(screen.contains("models/gemma_base"), "{screen}");
        assert!(screen.contains(tr("журнал")), "{screen}");
    }

    #[test]
    fn renders_paused_and_error_states() {
        let mut app = App::new();
        app.paused = true;
        app.stats.error = Some("boom".into());
        let screen = render(&app, 100, 16);
        assert!(screen.contains(tr("ПАУЗА")), "{screen}");
        assert!(screen.contains(tr("ОШИБКА")), "{screen}");
    }

    #[test]
    fn log_keeps_only_last_lines() {
        let mut app = App::new();
        for i in 0..100 {
            app.push_log(format!("line {i}"));
        }
        assert_eq!(app.log.len(), LOG_LIMIT.min(100));
        assert_eq!(app.log.back().unwrap(), "line 99");
    }
}
