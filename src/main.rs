fn main() {
    let iterations: i32 = 10000; // Number of iterations for training
    let learning_rate: f64 = 0.0125; // Learning rate for gradient descent

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

    let train_data_len = train_data.len();
    let mut w = 1.0; // Weight
    let mut b = 1.0; // Bias
    
    for _ in 0..iterations {

        w -= learning_rate * train_data.iter().fold(0.0, |acc, &(x, y)| {
            acc + 2.0 * x * ((w * x + b) - y)
        }) / train_data_len as f64;

        b -= learning_rate * train_data.iter().fold(0.0, |acc, &(x, y)| {
            acc + 2.0 * ((w * x + b) - y)
        }) / train_data_len as f64;
    }

    println!("Final weight: {}", w);
    println!("Final bias: {}", b);
}
