use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Embedding, EmbeddingConfig, Initializer, Linear, LinearConfig,
        Tanh,
    },
    prelude::*,
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    embedding: Embedding<B>,
    linear1: [Linear<B>; 3],
    bn1: [BatchNorm<B, 1>; 3],
    tanh1: [Tanh; 3],
    linear2: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let mut x = self.embedding.forward(input); // [B, 8, 10]

        // Explicit split of the embedding into even and odd indices
        // let even_indices = Tensor::<B, 1, Int>::from_data([0, 2, 4, 6], &device);
        // let odd_indices = Tensor::<B, 1, Int>::from_data([1, 3, 5, 7], &device);
        // let embedding_even = embedding.clone().select(1, even_indices);
        // let embedding_odd = embedding.clone().select(1, odd_indices);
        // let result = Tensor::cat(vec![embedding_even, embedding_odd], 2);
        // println!("result: {:?}", result.shape());

        // At each step, combine pairs of sequence elements and apply the corresponding layers
        for i in 0..3 {
            let shape = x.shape();
            let b = shape.dims[0] as i32;
            let seq = shape.dims[1] as i32;
            let emb = shape.dims[2] as i32;
            // Combine pairs: [B, seq, emb] -> [B, seq/2, emb*2]
            let x_reshaped = x.reshape([b, seq / 2, emb * 2]);
            let x_linear = self.linear1[i].forward(x_reshaped);
            let x_bn = self.bn1[i].forward(x_linear);
            let x_tanh = self.tanh1[i].forward(x_bn);
            x = x_tanh;
        }
        let x_tanh = x.squeeze::<2>(1);
        self.linear2.forward(x_tanh) // [B, 27]
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            embedding: EmbeddingConfig::new(27, 10).init(device),
            linear1: [
                LinearConfig::new(20, 200).init(device),
                LinearConfig::new(400, 200).init(device),
                LinearConfig::new(400, 200).init(device),
            ],
            bn1: [
                BatchNormConfig::new(4).init(device),
                BatchNormConfig::new(2).init(device),
                BatchNormConfig::new(1).init(device),
            ],
            tanh1: [Tanh::new(), Tanh::new(), Tanh::new()],
            linear2: LinearConfig::new(200, 27)
                .with_initializer(Initializer::KaimingNormal {
                    gain: 5.0 / 3.0,
                    fan_out_only: false,
                })
                .init(device),
        }
    }
}
