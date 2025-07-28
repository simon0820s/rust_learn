use crate::utils::prints::{print_early_stopping, print_train_progress_bar, print_train_results};
use std::io::{self, Write};
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

    pub fn train(
        &mut self,
        train_data: &Vec<(Vec<f64>, f64)>,
        epochs: usize,
        learning_rate: f64,
        patience: Option<usize>,
        accuracy_threshold: Option<f64>,
    ) {
        // Initialize progress bar and timer
        let bar = print_train_progress_bar(epochs);
        let time = Instant::now();

        // Early stopping
        let mut best_error = f64::MAX;
        let mut epochs_without_improvement = 0;
        let mut early_stopping = false;

        for epoch in 0..epochs {
            bar.inc(1);
            let mut grad_w = vec![0.0; self.w.len()];
            let mut grad_b = 0.0;
            let mut total_error = 0.0;
            let mut correct_evaluations = 0;

            for (x, y) in train_data {
                let y_pred = self.evaluate(x);
                let error = y_pred - y;
                total_error += error.abs();
                if let Some(threshold) = accuracy_threshold {
                    if error.abs() < threshold {
                        correct_evaluations += 1;
                    }
                }
                for i in 0..self.w.len() {
                    grad_w[i] += error * x[i];
                }
                grad_b += error;
            }

            for i in 0..self.w.len() {
                self.w[i] -= learning_rate * grad_w[i] / train_data.len() as f64;
            }
            self.b -= learning_rate * grad_b / train_data.len() as f64;

            if let Some(patience) = patience {
                let current_error = self.test_error(&train_data);
                if current_error < best_error {
                    best_error = current_error;
                    epochs_without_improvement = 0;
                } else {
                    epochs_without_improvement += 1;
                    if epochs_without_improvement >= patience {
                        print_early_stopping(
                            epoch,
                            &format!(
                                "error ({:.6}) does not improve at {epochs_without_improvement} epochs",
                                current_error
                            ),
                        );
                        early_stopping = true;
                        break;
                    }
                }
            }
            let avg_error = total_error / train_data.len() as f64;

            if let Some(_) = accuracy_threshold {
                let accuracy = (correct_evaluations as f64 / train_data.len() as f64) * 100.0;
                bar.set_message(format!(
                    "Error: {:.6} | Accuracy: {:.2}%",
                    avg_error, accuracy
                ));
            } else {
                bar.set_message(format!("Error: {:.6}", avg_error));
            }
        }

        if !early_stopping {
            bar.finish();
        }

        let final_error = self.test_error(train_data);

        print_train_results(time.elapsed().as_millis() as usize, final_error);
    }

    pub fn test_error(&self, test_data: &Vec<(Vec<f64>, f64)>) -> f64 {
        let mut me = 0.0;
        for (x, y) in test_data {
            let prediction = self.evaluate(x);
            me += (prediction - y).abs();
        }
        me / test_data.len() as f64
    }
    pub fn test_accuracy(&self, test_data: &Vec<(Vec<f64>, f64)>, treshold: f64) -> f64 {
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

    pub fn print_params(&self) {
        println!("Weights: {:?}", self.w);
        println!("Bias: {}", self.b);
    }
}
