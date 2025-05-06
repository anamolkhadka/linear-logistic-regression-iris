import numpy as np
from LinearRegression import LinearRegression


# Load origin model parameters
original_model = LinearRegression()
original_model.load("model_regression1.npz")
original_weights = original_model.weights
original_bias = original_model.bias

# Load regularized model parameters
regularized_model = LinearRegression()
regularized_model.load("model_regression1_reg.npz")
regularized_weights = regularized_model.weights
regularized_bias = regularized_model.bias

# Compute difference in weights and bias
weight_diff = np.abs(original_weights - regularized_weights)
bias_diff = np.abs(original_bias - regularized_bias)

# Print Results
print("Original Model Weights:", original_weights)
print("Regularized Model Weights:", regularized_weights)
print("Difference in Weights:", weight_diff)
print("\nOriginal Bias:", original_bias)
print("Regularized Bias:", regularized_bias)
print("Difference in Bias:", bias_diff)