fn main() {
    println!("Hello, world!");
}

#[cfg(test)]
mod tests {
    use burn::tensor::Tensor;
    use burn::{backend::autodiff::Autodiff};
    use burn_ndarray::{NdArray, NdArrayDevice};

    // I treated Burn as a blackbox and then validated the backprop on some of the operations

    #[test]
    fn test_loss_gradient() {
        // Assert that gradient of the loss is -1/N
        type B = Autodiff<NdArray<f32, i64, i8>>;
        let device: NdArrayDevice = Default::default();
        let tensor_in: Tensor<B, 1> = Tensor::from_data([0.7, 0.2, 0.1], &device).require_grad();
        let logprobs = tensor_in.clone().log();
        let n = 3.0;
        let loss = -logprobs.mean();
        let backprop = loss.backward();
        let grad: Vec<f32> = tensor_in
            .grad(&backprop)
            .unwrap()
            .to_data()
            .to_vec()
            .unwrap();
        let expected = vec![-1.0 / (n * 0.7), -1.0 / (n * 0.2), -1.0 / (n * 0.1)];
        for (g, e) in grad.iter().zip(expected.iter()) {
            assert!(
                (g - e).abs() < 1e-5,
                "Gradient {:?} does not match expected {:?}",
                g,
                e
            );
        }
    }

    #[test]
    fn test_sum_derivative() {
        // Assert that the derivative of the sum is 1
        type B = Autodiff<NdArray<f32, i64, i8>>;
        let device: NdArrayDevice = Default::default();
        let tensor_in: Tensor<B, 2> =
            Tensor::from_data([[2, 3, 4], [1, 2, 3]], &device).require_grad();
        let sum = tensor_in.clone().sum_dim(1);
        let grad = sum.backward();
        let grad_data: Vec<f32> = tensor_in.grad(&grad).unwrap().to_data().to_vec().unwrap();
        assert_eq!(grad_data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_matmul_derivative() {
        // Let Y = A @ B + C
        // Then: 
        // dY/dA = B^T (applied properly via chain rule as grad_output @ B^T)
        // dY/dB = A^T (applied via A^T @ grad_output)
        // dY/dC = Identity (just passes through gradient)

        type B = Autodiff<NdArray<f32, i64, i8>>;
        let device: NdArrayDevice = Default::default();
        let a: Tensor<B, 2> = Tensor::from_data([[1, 2], [3, 4]], &device).require_grad();   
        let b: Tensor<B, 2> = Tensor::from_data([[5, 6], [7, 8]], &device).require_grad();
        let c: Tensor<B, 2> = Tensor::from_data([[2, 2], [2, 2]], &device).require_grad();
        let ab = a.clone().matmul(b.clone());
        let abc = ab.clone().add(c.clone());
        let grad = abc.backward();
        let grad_a: Vec<f32> = a.grad(&grad).unwrap().to_data().to_vec().unwrap();
        let grad_b: Vec<f32> = b.grad(&grad).unwrap().to_data().to_vec().unwrap();
        let grad_c: Vec<f32> = c.grad(&grad).unwrap().to_data().to_vec().unwrap();

        // Check dY/dA
        let c_grad_a = Tensor::from_data([[1.0, 1.0], [1.0, 1.0]], &device);
        let c_grad_a = c_grad_a.clone().matmul(b.clone().transpose());
        let c_grad_a: Vec<f32> = c_grad_a.to_data().to_vec().unwrap();
        assert_eq!(grad_a, c_grad_a);

        // Check dY/dB
        let c_grad_b = Tensor::from_data([[1.0, 1.0], [1.0, 1.0]], &device);
        let c_grad_b = a.clone().transpose().matmul(c_grad_b);
        let c_grad_b: Vec<f32> = c_grad_b.to_data().to_vec().unwrap();
        assert_eq!(grad_b, c_grad_b);

        // Check dY/dC
        assert_eq!(grad_c, vec![1.0, 1.0, 1.0, 1.0]);
    }

    // I was too lazy to implement the rest and Burn doesn't have retain_grad so it is hard to check BatchNorm end to end
}
