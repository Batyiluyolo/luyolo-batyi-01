mod data;
mod model;
mod trainings;

use data::Dataset;
use textplots::{Chart, Plot, Shape};  // Added Plot back

fn main() {
    println!("\n=== Statistical Analysis Program ===\n");

    // Initialize dataset
    let dataset = Dataset::new();

    // Display data points
    println!("\nData Points:");
    for i in 0..dataset.len() {
        println!("Entry {}: input = {:.2}, output = {:.2}",
                 i + 1,
                 dataset.input_values[i],
                 dataset.output_values[i]);
    }

    // Calculate statistical measures
    let (input_mean, output_mean) = dataset.calculate_means();
    println!("\nStatistical Measures:");
    println!("Input Average: {:.3}", input_mean);
    println!("Output Average: {:.3}", output_mean);

    // Calculate regression parameters
    let n = dataset.len() as f64;
    let sum_input: f64 = dataset.input_values.iter().sum();
    let sum_output: f64 = dataset.output_values.iter().sum();
    let sum_products: f64 = dataset.input_values.iter()
        .zip(dataset.output_values.iter())
        .map(|(x, y)| x * y)
        .sum();
    let sum_squares: f64 = dataset.input_values.iter()
        .map(|x| x * x)
        .sum();

    let gradient = (n * sum_products - sum_input * sum_output)
        / (n * sum_squares - sum_input * sum_input);
    let intercept = (sum_output - gradient * sum_input) / n;

    println!("\nRegression Analysis:");
    println!("Mathematical Model: output = {:.3} × input + {:.3}", gradient, intercept);
    println!("Expected Model: output = 3.000 × input - 2.000");

    println!("\n=== Analysis Complete ===\n");
}
