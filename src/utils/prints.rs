use colored::*;
use indicatif::{ProgressBar, ProgressStyle};

pub fn print_train_progress_bar(epochs: usize) -> ProgressBar {
    let bar = ProgressBar::new(epochs as u64);
    bar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] [{bar:40.green}] {pos}/{len} ({percent}%)")
            .unwrap()
            .progress_chars("# "),
    );
    bar
}
pub fn print_train_results(duration: usize, error: f64) {
    println!(
        "Train completed in {}, final error: {}",
        format!("{}ms", duration).green(),
        format!("{:.4}", error).red()
    );
}

pub fn print_early_stopping(epoch: usize, error: f64) {
    println!(
        "Early stopping at epoch {}, with error: {}",
        epoch.to_string().yellow(),
        format!("{:.4}", error).red()
    );
}
