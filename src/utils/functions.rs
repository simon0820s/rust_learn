pub fn normalize_data(
    data: &Vec<(Vec<f64>, f64)>,
) -> (
    Vec<(Vec<f64>, f64)>, // normalized data
    Vec<f64>, // x_means
    Vec<f64>, // x_stds
    f64,      // y_mean
    f64,      // y_std
) {
    let input_dim = data[0].0.len();
    let n = data.len() as f64;

    let mut x_means = vec![0.0; input_dim];
    let mut x_stds = vec![0.0; input_dim];
    let mut y_mean = 0.0;
    let mut y_std = 0.0;

    for (x, y) in data.iter() {
        for i in 0..input_dim {
            x_means[i] += x[i];
        }
        y_mean += y;
    }

    for i in 0..input_dim {
        x_means[i] /= n;
    }
    y_mean /= n;

    for (x, y) in data.iter() {
        for i in 0..input_dim {
            x_stds[i] += (x[i] - x_means[i]).powi(2);
        }
        y_std += (y - y_mean).powi(2);
    }

    for i in 0..input_dim {
        x_stds[i] = (x_stds[i] / n).sqrt();
        if x_stds[i] == 0.0 {
            x_stds[i] = 1.0; // evita división por 0
        }
    }
    y_std = (y_std / n).sqrt();
    if y_std == 0.0 {
        y_std = 1.0;
    }

    let normalized = data
        .iter()
        .map(|(x, y)| {
            let norm_x = x
                .iter()
                .enumerate()
                .map(|(i, xi)| (xi - x_means[i]) / x_stds[i])
                .collect::<Vec<_>>();
            let norm_y = (y - y_mean) / y_std;
            (norm_x, norm_y)
        })
        .collect();

    (normalized, x_means, x_stds, y_mean, y_std)
}
