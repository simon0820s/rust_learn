use crate::utils::prints::{print_early_stopping, print_train_progress_bar, print_train_results};
use std::time::Instant;

pub struct LinearRegression {
    w: Vec<f64>,
    b: f64,
}

impl LinearRegression {
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
        x.iter()
            .zip(&self.w)
            .fold(self.b, |acc, (xi, wi)| acc + xi * wi)
    }

    pub fn train(
        &mut self,
        train_data: &Vec<(Vec<f64>, f64)>,
        epochs: usize,
        learning_rate: f64,
        batch_size: Option<usize>,
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
            let mut correct_evaluations = 0;

            let batches = self.get_batches(train_data, batch_size);

            batches.iter().for_each(|batche| {
                batche.iter().for_each(|(x, y)| {
                    let error = self.evaluate(x) - y;
                    if let Some(threshold) = accuracy_threshold {
                        if error.abs() < threshold {
                            correct_evaluations += 1;
                        }
                    }
                    for (grad_w, x) in grad_w.iter_mut().zip(x) {
                        *grad_w += error * x;
                    }
                    grad_b += error;
                });
                for (w, grad) in self.w.iter_mut().zip(&grad_w) {
                    *w -= learning_rate * grad / train_data.len() as f64;
                }

                self.b -= learning_rate * grad_b / train_data.len() as f64;
            });

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
            let avg_error = self.test_error(train_data);

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
        test_data
            .iter()
            .map(|(x, y)| (self.evaluate(x) - y).abs())
            .sum::<f64>()
            / test_data.len() as f64
    }
    pub fn test_accuracy(&self, test_data: &Vec<(Vec<f64>, f64)>, treshold: f64) -> f64 {
        test_data
            .iter()
            .map(|(x, y)| (self.evaluate(x) - y).abs() < treshold)
            .count() as f64
            / test_data.len() as f64
            * 100.0
    }

    pub fn summary(&self) {
        println!("╔═════════════════════════════════════════════╗");
        println!("║ {:<20} │ {:<20} ║", "Component", "Value");
        println!("╟─────────────────────────────────────────────╢");
        println!("║ {:<20} │ {:<20} ║", "Model type", "LinearRegression");
        println!("║ {:<20} │ {:<20} ║", "Input dimension", "1");
        println!("║ {:<20} │ {:<20} ║", "Output dimension", "1");
        println!("║ {:<20} │ {:<20} ║", "Trainable params", self.w.len() + 1);
        println!("╚═════════════════════════════════════════════╝");
    }

    pub fn print_params(&self) {
        println!("Weights: {:?}", self.w);
        println!("Bias: {}", self.b);
    }

    fn get_batches(
        &self,
        train_data: &Vec<(Vec<f64>, f64)>,
        batch_size: Option<usize>,
    ) -> Vec<Vec<(Vec<f64>, f64)>> {
        if let Some(size) = batch_size {
            train_data
                .chunks(size)
                .map(|chunk| chunk.to_vec())
                .collect()
        } else {
            vec![train_data.clone()]
        }
    }
}
