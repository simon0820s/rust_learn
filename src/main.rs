fn main() {

    let train_data: Vec<(i32, i32)> = vec![
        (0, 4),
        (1, 7),
        (2, 10),
        (3, 13),
        (4, 16),
        (5, 19),
        (6, 22),
        (7, 25),
        (8, 28),
        (9, 31),
    ];

    let b = 1;
    let m = 1;
    let mut total_error = 0;

    // Get error by point 
    train_data.iter().for_each(|(x, y)| {
        println!("x: {}, y: {}", x, y);
        total_error += (y - (m * x + b)).pow(2).abs();
    });

    // Calculate mean squared error
    let mean_squared_error = total_error as f64 / train_data.len() as f64;

}
