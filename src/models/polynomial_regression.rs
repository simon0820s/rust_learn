use crate::utils::functions::get_multivariable_polynomial_exponent_combination;
use crate::utils::prints::{print_early_stopping, print_train_progress_bar};
use rand::distributions::{Distribution, Uniform};
use std::time::Instant;

pub struct PolynomialRegression {
    w: Vec<f64>,
    degrees_combinations: Vec<Vec<usize>>,
}

impl PolynomialRegression {
    pub fn new(variables: usize, degree: usize) -> Self {
        let multivariable_polynomial_exponent_combinations =
            get_multivariable_polynomial_exponent_combination(variables, degree);
        let num_weights = multivariable_polynomial_exponent_combinations.len();

        let mut rng = rand::thread_rng();
        let uniform = Uniform::from(-1.0..=1.0);
        let w: Vec<f64> = (0..num_weights).map(|_| uniform.sample(&mut rng)).collect();

        PolynomialRegression {
            w,
            degrees_combinations: multivariable_polynomial_exponent_combinations,
        }
    }

    pub fn evaluate(&self, x: &Vec<f64>) -> f64 {
        let mut acc = 0.0;
        for i in 0..self.w.len() {
            let mut multiplication = 1.0;
            for j in 0..self.degrees_combinations[i].len() {
                multiplication *= x[j].powi(self.degrees_combinations[i][j] as i32);
            }
            acc += self.w[i] * multiplication;
        }
        acc
    }

    pub fn train(
        &mut self,
        train_data: &Vec<(Vec<f64>, f64)>,
        epochs: usize,
        learning_rate: f64,
        patience: Option<usize>,
    ) {
        assert!(!train_data.is_empty(), "train_data is empty");
        assert_eq!(
            train_data[0].0.len(),
            self.degrees_combinations[0].len(),
            "Input dimensions do not match"
        );

        let bar = print_train_progress_bar(epochs);
        let time = Instant::now();

        // Early stopping
        let mut best_error = f64::MAX;
        let mut epochs_without_improvement = 0;
        let mut early_stopping = false;

        for epoch in 0..epochs {
            bar.inc(1);
            let mut grad_w = vec![0.0; self.w.len()];

            for (x, y) in train_data {
                let error = self.evaluate(x) - y;
                for i in 0..self.w.len() {
                    let mut multiplication = 1.0;
                    for j in 0..self.degrees_combinations[i].len() {
                        multiplication *= x[j].powi(self.degrees_combinations[i][j] as i32);
                    }
                    grad_w[i] += 2.0 * error * multiplication;
                }
            }
            for i in 0..self.w.len() {
                self.w[i] -= learning_rate * grad_w[i] / train_data.len() as f64;
            }
            if let Some(patience) = patience {
                let current_error = self.test_error(&train_data);
                if current_error < best_error {
                    best_error = current_error;
                    epochs_without_improvement = 0;
                    println!("Best error:{:.8}", best_error);
                } else {
                    epochs_without_improvement += 1;
                    if epochs_without_improvement >= patience {
                        print_early_stopping(
                            epoch,
                            &format!(
                                "Error: {:.8} did not improve at {} epochs",
                                current_error, epochs_without_improvement
                            ),
                        );
                        early_stopping = true;
                        break;
                    }
                }
            }

            if self.w.iter().any(|&w| w.is_nan()) {
                print_early_stopping(epoch, "NaN Weights detected");
                early_stopping = true;
                break;
            }
        }
        if !early_stopping {
            bar.finish();
        }
    }

    pub fn test_error(&self, test_data: &Vec<(Vec<f64>, f64)>) -> f64 {
        let mut me = 0.0;
        for (x, y) in test_data {
            me += (self.evaluate(x) - y).abs();
        }
        me / test_data.len() as f64
    }

    // pub fn test_accuracy(&self, test_data: &Vec<(f64, f64)>, treshold: f64) -> f64 {
    //     let mut accuracy = 0.0;
    //     for (x, y) in test_data {
    //         if (self.evaluate(x) - y).abs() < treshold {
    //             accuracy += 1.0;
    //         }
    //     }
    //     (accuracy / test_data.len() as f64) * 100.0
    // }

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
        println!("Degrees combinations: {:?}", self.degrees_combinations);
    }
}
