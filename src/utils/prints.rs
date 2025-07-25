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
pub fn print_train_results(duration: usize) {
    println!("✅ Train completed in {}.", format!("{}ms", duration).green());
}
