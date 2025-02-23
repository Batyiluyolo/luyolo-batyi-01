use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Tensor, backend::Backend};

#[derive(Module, Debug)]
pub struct LinearRegression<B: Backend> {
    predictor: Linear<B>,
}

#[derive(Debug)]
pub struct ModelCoefficients {
    pub gradient: f64,
    pub offset: f64,
}

impl<B: Backend> LinearRegression<B> {
    pub fn initialize(device: &B::Device) -> Self {
        println!("INFO: Initializing prediction model");
        Self {
            predictor: LinearConfig::new(1, 1)
                .with_bias(true)
                .init_with_device(device),
        }
    }

    pub fn compute(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        self.predictor.forward(inputs)
    }

    pub fn extract_coefficients(&self) -> ModelCoefficients {
        let weights = self.predictor.weight().to_vec2();
        let bias = self.predictor.bias().unwrap().to_vec1();

        ModelCoefficients {
            gradient: weights[0][0] as f64,
            offset: bias[0] as f64,
        }
    }
}