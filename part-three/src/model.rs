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
    linear1: Linear<B>,
    bn: BatchNorm<B, 1>,
    tanh: Tanh,
    linear2: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 2> {
        let embedding = self.embedding.forward(input);
        let embedding = embedding.reshape([-1, 60]);
        let linear1 = self.linear1.forward(embedding);
        let linear1 = linear1.unsqueeze_dim::<3>(2);
        let bn = self.bn.forward(linear1);
        let tanh = self.tanh.forward(bn);
        let tanh = tanh.squeeze::<2>(2);
        self.linear2.forward(tanh)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            embedding: EmbeddingConfig::new(27, 20).init(device),
            linear1: LinearConfig::new(60, 200).init(device),
            bn: BatchNormConfig::new(200).init(device),
            tanh: Tanh::new(),
            linear2: LinearConfig::new(200, 27)
                .with_initializer(Initializer::KaimingNormal {
                    gain: 5.0 / 3.0,
                    fan_out_only: false,
                })
                .init(device),
        }
    }
}
