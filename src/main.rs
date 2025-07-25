mod models;
mod utils;
use crate::models::linear_regression::LinearRegression;

fn main() {
    let train_data: Vec<(f64, f64)> = vec![
        (0.0, 4.0),
        (1.0, 7.0),
        (2.0, 10.0),
        (3.0, 13.0),
        (4.0, 16.0),
        (5.0, 19.0),
        (6.0, 22.0),
        (7.0, 25.0),
        (8.0, 28.0),
        (9.0, 31.0),
        (10.0, 34.0),
    ];

    let mut linear_regression = LinearRegression::new();
    linear_regression.evaluate(5.0);
    linear_regression.train(train_data, 10000, 0.01);
    linear_regression.summary();
}
