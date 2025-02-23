# luyolo-batyi-01

# Linear Regression Model in Rust using Burn

Project Overview

This project implements a simple AI model for linear regression using Rust and the Burn library (version 0.16.0).
The model is designed to predict values based on the equation: y=2x+1 with synthetic data used for training.
The implementation follows the assignment requirements, including training the model and visualizing the results using the textplots crate.

Setup and Running the Project
Steps to Set Up the Project

Clone the Repository:

git clone https://github.com/Batyiluyolo/luyolo-batyi-01.git
cd linear_regression_model

Build the Project:
cargo build

Run the Model:
cargo run

**Known Issue: The Code is Not Running**

Currently, the program does not execute successfully due to errors in some of the code files. The issues appear to be related to:
* Tensor operations in trainings.rs causing type mismatches.
* Backend compatibility problems in the Burn library.
* Possible missing or incorrectly structured dependencies.
Despite multiple debugging attempts using AI tools and documentation, the errors persist. Further troubleshooting is needed to resolve these issues.

**Approach & Challenges**
_Approach_
Generating Synthetic Data:
* Implemented a dataset generator in data.rs with (x, y) pairs using the equation y=2x+1.
* Added noise to simulate real-world conditions.

_Model Definition:_
* Used the Burn library to define a linear regression model in model.rs.
* Implemented a forward pass for predictions.

_Training the Model:_
* Implemented an optimizer using the Adam algorithm in trainings.rs.
* Defined a loss function using Mean Squared Error (MSE).
* Trained the model using gradient descent.

_Evaluating & Plotting Results:_
* Visualized training results using the textplots crate.

**Challenges Faced**
* Understanding Burn's Backend Requirements: The burn library requires a compatible backend. Debugging Tensor and Backend errors took time.
* Error Handling in Training: Initially, errors occurred due to improper tensor operations in trainings.rs. Adjusting tensor dimensions and reshaping resolved these issues.
* Compilation Errors: Some features were incompatible due to mismatched dependency versions. Ensuring dependencies matched the required versions fixed this.
* AI-Assisted Debugging: Used ChatGPT and DeepSeek AI to troubleshoot syntax and logic errors.

# **Reflection on Learning**
**Assistance from AI & Documentation**
* AI tools (ChatGPT, DeepSeek AI) were used extensively for debugging Rust errors and clarifying Burn library usage.
* Official Rust and Burn documentation helped with module implementation.
* Online tutorials and YouTube videos provided insight into implementing linear regression.

**Key Learnings**
* Burn Library Usage: Gained experience in setting up and configuring a deep learning model using Burn.
* Rust Error Handling: Improved ability to debug Rust-related errors, especially regarding type mismatches and tensor operations.
* Working with AI Tools: Learned to refine AI-generated code and cross-check against documentation.

**Unresolved Issues**
Despite following the correct approach, I encountered persistent errors in tensor operations. While some issues were resolved, others require further debugging. The main challenges include:
* Ensuring proper tensor shape compatibility during training.
* Managing device compatibility for Burn’s backend.
* Addressing any hidden compilation issues that prevent execution.

**Resources Used**
* Rust Documentation: https://doc.rust-lang.org/
* Burn Library Docs: https://docs.rs/burn/0.16.0/burn/
* YouTube & AI Tools: ChatGPT, DeepSeek AI for debugging and optimization strategies.

