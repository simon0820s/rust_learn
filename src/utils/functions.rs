pub fn get_multivariable_polynomial_exponent_combination(n: usize, degree: usize) -> Vec<Vec<usize>> {
    fn helper(n: usize, degree: usize, current: Vec<usize>, acc: &mut Vec<Vec<usize>>) {
        if current.len() == n {
            acc.push(current);
            return;
        }
        for i in 0..=degree {
            let mut next = current.clone();
            next.push(i);
            helper(n, degree, next, acc);
        }
    }

    let mut result = Vec::new();
    helper(n, degree, Vec::new(), &mut result);
    result
}