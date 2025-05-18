mod model;

use burn::LearningRate;
use burn::backend::Autodiff;
use burn::module::AutodiffModule;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::GradientsParams;
use burn::optim::{AdamConfig, Optimizer};
use burn::prelude::Backend;
use burn::tensor::Int;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, activation::softmax};
use burn_ndarray::NdArray;
use model::{Model, ModelConfig};
use rand::Rng;
use std::fs::File;
use std::io::{BufReader, Read};

fn stoi(c: char) -> u32 {
    if c == '.' { 0 } else { c as u32 - 96 }
}

fn itos(i: u32) -> char {
    if i == 0 { '.' } else { (i + 96) as u8 as char }
}

/// Load the data as two arrays of encoded trigrams
/// Example: (e, m, m) -> (a)
fn load_data(path: &str) -> (Vec<u32>, Vec<u32>) {
    let f = File::open(path).unwrap();
    let mut reader = BufReader::new(f);
    let mut text = String::new();
    reader.read_to_string(&mut text).unwrap();
    let words: Vec<&str> = text.lines().filter(|line| !line.is_empty()).collect();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for word in words {
        let new_word = format!("{}.", word);
        let chars = new_word.chars();
        let mut context = [0, 0, 0];
        for curr in chars {
            let ix = stoi(curr);
            xs.extend(context);
            ys.push(ix);
            context = [context[1], context[2], ix];
        }
    }
    (xs, ys)
}

/// Draw a random sample from the probability vector
fn draw_sample<B: Backend>(tensor: &Tensor<B, 1>) -> usize {
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

/// Create batches from xs and ys
fn make_batches(xs: &[u32], ys: &[u32], batch_size: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut batches = Vec::new();
    let num_batches = ys.len() / batch_size;
    for i in 0..num_batches {
        let start_y = i * batch_size;
        let end_y = start_y + batch_size;
        let ys_batch = ys[start_y..end_y].to_vec();
        let start_x = start_y * 3;
        let end_x = end_y * 3;
        let xs_batch = xs[start_x..end_x].to_vec();
        batches.push((xs_batch, ys_batch));
    }
    batches
}

fn train<B: AutodiffBackend>(device: B::Device) -> Model<B> {
    let config_optimizer = AdamConfig::new();
    let (xs, ys) = load_data("names.txt");
    let optimizer = config_optimizer.init::<B, Model<B>>();
    let batch_size = 256;
    let batches = make_batches(&xs, &ys, batch_size);
    let config_model = ModelConfig {};
    let mut model = config_model.init::<B>(&device);
    let lr: LearningRate = 0.001;
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    for i in 0..50 {
        let mut total_loss = 0.0;
        for (xs_batch, ys_batch) in &batches {
            let input_tensor: Tensor<B, 1, Int> =
                Tensor::<B, 1, Int>::from_data(xs_batch.as_slice(), &device);
            let input_tensor = input_tensor.unsqueeze_dim::<2>(1);
            let input_tensor: Tensor<B, 2, Int> = input_tensor.reshape([-1, 3]);
            let logits = model.forward(input_tensor.clone());
            let output_tensor = Tensor::<B, 1, Int>::from_data(ys_batch.as_slice(), &device);
            let loss = loss_fn.forward(logits, output_tensor);
            total_loss += loss.to_data().as_slice::<f32>().unwrap()[0];
            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optimizer.clone().step(lr, model, grads);
        }
        let loss = total_loss / batches.len() as f32;
        println!("iteration: {}, loss: {}", i, loss);
    }
    model
}

/// Run inference on some examples
pub fn inference<B: Backend>(model: &Model<B>, device: B::Device) {
    for _ in 0..5 {
        let mut input = [0, 0, 0];
        let mut name: Vec<char> = vec![];
        loop {
            let input_tensor = Tensor::<B, 1, Int>::from_data(input.as_slice(), &device);
            let input_tensor = input_tensor.unsqueeze_dim::<2>(1);
            let input_tensor: Tensor<B, 2, Int> = input_tensor.reshape([-1, 3]);
            let out = model.forward(input_tensor);
            let out = softmax(out, 1).squeeze(0);
            let idx = draw_sample(&out);
            if idx == 0 {
                break;
            }
            let c = itos(idx as u32);
            name.push(c);
            input = [input[1], input[2], idx];
        }
        println!("{}", name.iter().collect::<String>());
    }
}

fn main() {
    env_logger::init();
    let device = Default::default();
    let model = train::<Autodiff<NdArray>>(device);
    inference(&model.valid(), device);
}
