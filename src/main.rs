mod models;
mod utils;
use crate::models::linear_regression::LinearRegression;

fn main() {
    let train_data: Vec<(Vec<f64>, f64)> = vec![
        (vec![0.0, 0.0], 10.0),
        (vec![1.0, 1.0], 15.0),
        (vec![1.0, 0.0], 12.0),
        (vec![0.0, 1.0], 13.0),
    ];

    let mut linear_regression = LinearRegression::new(2);
    linear_regression.train(train_data, 1000000, 1e-4);
    println!("{}", linear_regression.evaluate(&vec![11.0, 11.0])); // 65
    linear_regression.summary();
}
