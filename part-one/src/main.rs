mod model;

use burn::LearningRate;
use burn::backend::Autodiff;
use burn::optim::GradientsParams;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, activation::softmax};
use model::{Model, ModelConfig};
use rand::Rng;
use std::fs::File;
use std::io::{BufReader, Read};

#[cfg(not(feature = "metal"))]
use burn_ndarray::NdArray;

#[cfg(feature = "metal")]
use burn::backend::Metal;

fn stoi(c: char) -> u32 {
    if c == '.' { 0 } else { c as u32 - 96 }
}

fn itos(i: u32) -> char {
    if i == 0 { '.' } else { (i + 96) as u8 as char }
}

/// Load the data as two arrays of encoded bigrams
fn load_data(path: &str) -> (Vec<u32>, Vec<u32>) {
    let f = File::open(path).unwrap();
    let mut reader = BufReader::new(f);
    let mut text = String::new();
    reader.read_to_string(&mut text).unwrap();
    let words: Vec<&str> = text.lines().filter(|line| !line.is_empty()).collect();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for word in words {
        let new_word = format!(".{}.", word);
        let mut chars = new_word.chars();
        let mut prev = chars.next().unwrap();
        for curr in chars {
            xs.push(stoi(prev));
            ys.push(stoi(curr));
            prev = curr;
        }
    }
    (xs, ys)
}

/// Draw a random sample from the probability vector
pub fn draw_sample<B: Backend>(tensor: &Tensor<B, 1>) -> usize {
    let data = tensor.to_data();
    let data = data.as_slice::<f32>().unwrap();
    let sum: f32 = data.iter().sum();
    let normalized: Vec<f32> = data.iter().map(|x| x / sum).collect();

    let mut cdf = Vec::with_capacity(normalized.len());
    let mut acc = 0.0;
    for &x in &normalized {
        acc += x;
        cdf.push(acc);
    }

    let mut rng = rand::rng();
    let r: f32 = rng.random_range(0.0..1.0);
    cdf.iter().position(|&x| x > r).unwrap_or(0)
}

pub fn run<B: AutodiffBackend>(device: B::Device) {
    let config_model = ModelConfig {};
    let config_optimizer = AdamConfig::new();
    let (xs, ys) = load_data("names.txt");
    let mut model = config_model.init::<B>(&device);
    let optimizer = config_optimizer.init::<B, Model<B>>();
    let input_tensor: Tensor<B, 1> = Tensor::<B, 1>::from_data(xs.as_slice(), &device);
    let row_indices: Tensor<B, 1, burn::tensor::Int> =
        Tensor::arange(0..input_tensor.shape().num_elements() as i64, &device);
    let lr: LearningRate = 0.1;

    for i in 0..100 {
        let logits = model.forward(input_tensor.clone());
        let probs = softmax(logits, 1);
        let row = probs.select(0, row_indices.clone());
        let indices = Tensor::<B, 1, burn::tensor::Int>::from_data(ys.as_slice(), &device)
            .unsqueeze_dim::<2>(1);
        let selected = row.gather(1, indices);
        let loss = -selected.log().mean();
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.clone().step(lr, model, grads);
        println!("iteration: {}, loss: {}", i, loss.to_data());
    }

    // Inference example
    for _ in 0..5 {
        let mut ix = 0;
        let mut name: Vec<char> = vec![];
        loop {
            let input = Tensor::<B, 1>::from_data([ix], &device);
            let out = model.forward(input);
            let out = softmax(out, 1).squeeze(0);
            let idx = draw_sample(&out);
            if idx == 0 {
                break;
            }
            let c = itos(idx as u32);
            name.push(c);
            ix = idx;
        }
        println!("{}", name.iter().collect::<String>());
    }
}

#[cfg(feature = "metal")]
fn main() {
    let device = Default::default();
    run::<Autodiff<Metal>>(device);
}

#[cfg(not(feature = "metal"))]
fn main() {
    let device = Default::default();
    run::<Autodiff<NdArray>>(device);
}
