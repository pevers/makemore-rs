use burn::{
    nn::{Linear, LinearConfig},
    prelude::*,
    tensor::{Tensor, backend::Backend},
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    linear1: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 2> {
        let one_hot = input.one_hot(27);
        self.linear1.forward(one_hot)
    }
}

#[derive(Config, Debug)]
pub struct ModelConfig {}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(27, 27).init(device),
        }
    }
}
