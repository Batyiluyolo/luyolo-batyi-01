use burn::tensor::{backend::Backend, Tensor};
use burn::optim::{Adam, Optimizer};
use crate::model::{LinearRegression, ModelCoefficients};
use crate::data::Dataset;

pub struct TrainingManager<B: Backend> {
    prediction_model: LinearRegression<B>,
    optimizer: Adam<B>,
    learning_rate: f64,
}

impl<B: Backend> TrainingManager<B> {
    pub fn initialize(device: &B::Device, learning_rate: f64) -> Self {
        Self {
            prediction_model: LinearRegression::initialize(device),
            optimizer: Adam::new(learning_rate as f32),
            learning_rate,
        }
    }

    pub fn execute_training_step(&mut self, data: &Dataset<B>) -> f64 {
        let batch_count = data.input_values.shape()[0];
        let inputs = data.input_values.clone().reshape([batch_count, 1]);
        let targets = data.output_values.clone().reshape([batch_count, 1]);

        let predictions = self.prediction_model.compute(inputs);
        let error = self.calculate_error(&predictions, &targets);

        self.optimizer.zero_grad();
        error.backward();
        self.optimizer.step();

        error.to_scalar() as f64
    }

    pub fn get_model_coefficients(&self) -> ModelCoefficients {
        self.prediction_model.extract_coefficients()
    }

    fn calculate_error(&self, predictions: &Tensor<B, 2>, targets: &Tensor<B, 2>) -> Tensor<B, 0> {
        let differences = predictions - targets;
        (differences.clone() * differences).mean()
    }

    pub fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }
}
