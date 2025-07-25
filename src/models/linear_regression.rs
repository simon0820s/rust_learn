use crate::utils::prints::{print_train_progress_bar, print_train_results};
use std::time::Instant;

pub struct LinearRegression {
    pub w: f64,
    pub b: f64,
}

impl LinearRegression {
    // Create new instance with default parameters.
    pub fn new() -> Self {
        LinearRegression { w: 0.0, b: 0.0 }
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        self.w * x + self.b
    }

    pub fn train(&mut self, train_data: Vec<(f64, f64)>, epochs: usize, learning_rate: f64) {
        let bar = print_train_progress_bar(epochs);
        let time = Instant::now();

        let train_data_len = train_data.len();

        for _ in 0..epochs {
            bar.inc(1);
            self.w -= learning_rate
                * train_data.iter().fold(0.0, |acc, &(x, y)| {
                    acc + 2.0 * x * ((self.w * x + self.b) - y)
                })
                / train_data_len as f64;

            self.b -= learning_rate
                * train_data
                    .iter()
                    .fold(0.0, |acc, &(x, y)| acc + 2.0 * ((self.w * x + self.b) - y))
                / train_data_len as f64;
        }
        bar.finish();
        print_train_results(time.elapsed().as_millis() as usize);
    }

    pub fn summary(&self) {
        println!("╔═════════════════════════════════════════════╗");
        println!("║ {:<20} │ {:<20} ║", "Component", "Value");
        println!("╟─────────────────────────────────────────────╢");
        println!("║ {:<20} │ {:<20} ║", "Model type", "LinearRegression");
        println!("║ {:<20} │ {:<20} ║", "Input dimension", "1");
        println!("║ {:<20} │ {:<20} ║", "Output dimension", "1");
        println!("║ {:<20} │ {:<20} ║", "Trainable parameters", "2");
        println!("║ {:<20} │ {:.4}{:<14} ║", "w", self.w, " ");
        println!("║ {:<20} │ {:.4}{:<14} ║", "b", self.b, " ");
        println!("╚═════════════════════════════════════════════╝");
    }
}
