pub struct Dataset {
    pub input_values: Vec<f64>,
    pub output_values: Vec<f64>,
}

impl Dataset {
    pub fn new() -> Self {
        // Different equation: y = 3x - 2
        let input_values = vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let output_values = vec![-5.0, -2.0, 1.0, 4.0, 7.0, 10.0];

        println!("INFO: Initializing dataset with {} entries", input_values.len());
        println!("Inputs: {:?}", input_values);
        println!("Outputs: {:?}", output_values);

        Dataset {
            input_values,
            output_values
        }
    }

    pub fn len(&self) -> usize {
        self.input_values.len()
    }

    pub fn calculate_means(&self) -> (f64, f64) {
        let input_mean = self.input_values.iter().sum::<f64>() / self.len() as f64;
        let output_mean = self.output_values.iter().sum::<f64>() / self.len() as f64;
        (input_mean, output_mean)
    }

    pub fn is_empty(&self) -> bool {
        self.input_values.is_empty()
    }
}
