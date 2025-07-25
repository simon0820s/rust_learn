use crate::utils::prints::{print_train_progress_bar, print_train_results};
use std::time::Instant;

pub struct LinearRegression {
    w: Vec<f64>,
    b: f64,
}

impl LinearRegression {
    // Create new instance with default parameters.
    pub fn new(input_lenght: usize) -> Self {
        LinearRegression {
            w: (0..input_lenght).map(|x| x as f64 * 0.0).collect(),
            b: 0.0,
        }
    }

    pub fn evaluate(&self, x: &Vec<f64>) -> f64 {
        if x.len() != self.w.len() {
            panic!(
                "Expected input vector with {} entries, got vector with {} entries",
                self.w.len(),
                x.len()
            );
        }
        let mut acc = 0.0;
        for i in 0..x.len() {
            acc += x[i] * self.w[i];
        }
        acc + self.b
    }

    pub fn train(&mut self, train_data: Vec<(Vec<f64>, f64)>, epochs: usize, learning_rate: f64) {
        let bar = print_train_progress_bar(epochs);
        let time = Instant::now();

        let train_data_len = train_data.len();

        for _ in 0..epochs {
            bar.inc(1);
            let mut grad_w = vec![0.0; self.w.len()];
            let mut grad_b = 0.0;

            for (x, y) in &train_data {
                let y_pred = self.evaluate(x);
                let error = y_pred - y;
                for i in 0..self.w.len() {
                    grad_w[i] += 2.0 * error * x[i];
                }
                grad_b += 2.0 * error;
            }

            for i in 0..self.w.len() {
                self.w[i] -= learning_rate * grad_w[i] / train_data.len() as f64;
            }
            self.b -= learning_rate * grad_b / train_data.len() as f64;
        }

        bar.finish();
        let mut mean_squared_error = 0.0;

        for (x, y) in &train_data {
            let prediction = self.evaluate(x);
            mean_squared_error += (prediction - y).powi(2);
        }
        mean_squared_error /= train_data_len as f64;

        print_train_results(time.elapsed().as_millis() as usize, mean_squared_error);
    }

    pub fn summary(&self) {
        println!("╔═════════════════════════════════════════════╗");
        println!("║ {:<20} │ {:<20} ║", "Component", "Value");
        println!("╟─────────────────────────────────────────────╢");
        println!("║ {:<20} │ {:<20} ║", "Model type", "LinearRegression");
        println!("║ {:<20} │ {:<20} ║", "Input dimension", "1");
        println!("║ {:<20} │ {:<20} ║", "Output dimension", "1");
        println!(
            "║ {:<20} │ {:<20} ║",
            "Trainable parameters",
            self.w.len() + 1
        );
        println!("╚═════════════════════════════════════════════╝");
    }
}
