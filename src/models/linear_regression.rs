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
        let train_data_len = train_data.len();
        for _ in 0..epochs{
            self.w -= learning_rate
                * train_data
                    .iter()
                    .fold(0.0, |acc, &(x, y)| acc + 2.0 * x * ((self.w * x + self.b) - y))
                / train_data_len as f64;

            self.b -= learning_rate
                * train_data
                    .iter()
                    .fold(0.0, |acc, &(x, y)| acc + 2.0 * ((self.w * x + self.b) - y))
                / train_data_len as f64;
        }
    }
}
