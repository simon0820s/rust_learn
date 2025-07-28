use crate::utils::prints::{print_train_progress_bar, print_train_results};
use std::time::Instant;

pub struct PolynomialRegression {
    w: Vec<f64>,
}

impl PolynomialRegression {
    // Create new instance with default parameters.
    pub fn new(degree: usize) -> Self {
        PolynomialRegression {
            w: (0..(degree + 1)).map(|x| x as f64 * 0.0).collect(),
        }
    }

    pub fn evaluate(&self, x: &f64) -> f64 {
        let mut acc = 0.0;
        for i in 0..self.w.len() {
            acc += self.w[i] * x.powi(i as i32);
        }
        acc
    }

    pub fn train(
        &mut self,
        train_data: &Vec<(f64, f64)>,
        epochs: usize,
        learning_rate: f64,
        patience: Option<usize>,
    ) {
        // Initialize progress bar and timer
        let bar = print_train_progress_bar(epochs);
        let time = Instant::now();

        // Early stopping
        let mut best_error = f64::MAX;
        let mut epochs_without_improvement = 0;
        let mut early_stopping = false;

        let train_data_len = train_data.len();

        for _ in 0..epochs {
            bar.inc(1);
            let mut grad_w = vec![0.0; self.w.len()];

            for i in 0..train_data_len {
                let error = self.evaluate(&train_data[i].0) - train_data[i].1;
                for j in 0..self.w.len() {
                    grad_w[j] += 2.0 as f64 * error * train_data[i].0.powi(j as i32);
                }
            }
            for i in 0..self.w.len() {
                self.w[i] -= learning_rate * grad_w[i] / train_data_len as f64;
            }

            if let Some(patience) = patience {
                let current_error = self.test_error(&train_data);
                if current_error < best_error {
                    best_error = current_error;
                    epochs_without_improvement = 0;
                } else {
                    epochs_without_improvement += 1;
                    if epochs_without_improvement >= patience {
                        early_stopping = true;
                        break;
                    }
                }
            }
        }

        if !early_stopping {
            bar.finish();
        }

        let mut mean_squared_error = 0.0;

        for (x, y) in train_data {
            let prediction = self.evaluate(x);
            mean_squared_error += (prediction - y).powi(2);
        }
        mean_squared_error /= train_data_len as f64;

        print_train_results(time.elapsed().as_millis() as usize, mean_squared_error);
    }

    pub fn test_error(&self, test_data: &Vec<(f64, f64)>) -> f64 {
        let mut me = 0.0;
        for (x, y) in test_data {
            let prediction = self.evaluate(x);
            me += (prediction - y).abs();
        }
        me / test_data.len() as f64
    }
    pub fn test_accuracy(&self, test_data: &Vec<(f64, f64)>, treshold: f64) -> f64 {
        let mut accuracy = 0.0;
        for (x, y) in test_data {
            if (self.evaluate(x) - y).abs() < treshold {
                accuracy += 1.0;
            }
        }
        (accuracy / test_data.len() as f64) * 100.0
    }

    pub fn summary(&self) {
        println!("╔═════════════════════════════════════════════╗");
        println!("║ {:<20} │ {:<20} ║", "Component", "Value");
        println!("╟─────────────────────────────────────────────╢");
        println!("║ {:<20} │ {:<20} ║", "Model type", "PolynomialRegression");
        println!("║ {:<20} │ {:<20} ║", "Input dimension", "1");
        println!("║ {:<20} │ {:<20} ║", "Output dimension", "1");
        println!("║ {:<20} │ {:<20} ║", "Trainable parameters", self.w.len());
        println!("╚═════════════════════════════════════════════╝");
    }

    pub fn print_params(&self) {
        println!("Weights: {:?}", self.w);
    }
}
