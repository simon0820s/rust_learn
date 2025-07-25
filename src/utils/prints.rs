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
pub fn print_train_results(
    w: f64,
    b: f64,
    epochs: usize,
    learning_rate: f64,
) -> String {
    format!(
        "Training completed with w: {:.4}, b: {:.4}, epochs: {}, learning rate: {:.4}",
        w, b, epochs, learning_rate
    )
}