use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{backend::Backend, Tensor},
};

#[derive(Module, Debug, Clone)] // Added Clone trait
struct Model<B: Backend> {
    weight: Linear<B>,
}

impl<B: Backend> Model<B> {
    fn new(device: &B::Device) -> Self {
        Self {
            weight: LinearConfig::new(1, 1).init(device), // Passed the device argument
        }
    }

    fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.weight.forward(input)
    }
}

fn mean_squared_error<B: Backend>(
    predictions: Tensor<B, 2>,
    targets: Tensor<B, 2>
) -> Tensor<B, 2> {
    (predictions - targets).powf(2.0).mean()
}

fn main() {
    let device = burn::tensor::backend::Default::default(); // Initialize the device
    let model = Model::new(&device);

    let x_values = [0.1, 0.2, 0.3, 0.4];
    let y_values = [0.2, 0.4, 0.6, 0.8];

    for (x, y) in x_values.iter().zip(y_values.iter()) {
        let x_tensor = Tensor::<burn::tensor::backend::Default, 2>::from_floats(&[[*x]], &device);
        let y_tensor = Tensor::<burn::tensor::backend::Default, 2>::from_floats(&[[*y]], &device);

        let prediction = model.forward(x_tensor);
        let loss = mean_squared_error(prediction, y_tensor);

        println!("Loss: {:?}", loss);
    }
}
