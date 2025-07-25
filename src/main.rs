mod models;
mod utils;
use crate::models::linear_regression::LinearRegression;

fn main() {
    let train_data: Vec<(Vec<f64>, f64)> = vec![
        (vec![0.0, 0.0], 2.0),
        (vec![1.0, 1.0], 8.0),
        (vec![1.0, 0.0], 4.0),
        (vec![0.0, 1.0], 6.0),
    ];

    let test_data: Vec<(Vec<f64>, f64)> = vec![
        (vec![10.0, 0.0], 22.0),
        (vec![5.0, 5.0], 32.0),
        (vec![10.0, 10.0], 62.0),
    ];

    let mut linear_regression = LinearRegression::new(2);
    linear_regression.train(train_data, 10000000, 1e-4, Some(5));
    println!("Error: {}", linear_regression.test_error(&test_data));
    println!(
        "Accuracy: {}%",
        linear_regression.test_accuracy(&test_data, 0.01)
    );
    linear_regression.print_params();
    linear_regression.summary();
}
