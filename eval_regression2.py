import numpy as np
from LinearRegression import LinearRegression

# Evaluation script for single-output regression (Model 2)

# Load test dataset
test_data = np.load('test_data.npz')
X_test = test_data['X_test'][:, 2:]  # Petal Length & Petal Width as input
y_test = test_data['X_test'][:, 0]   # Sepal Length as target

# Load trained model
model = LinearRegression()
model.load("model_regression2.npz")

# Make predictions
y_pred = model.predict(X_test)

# Compute Mean Squared Error (MSE)
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error on Test Data: {mse:.4f}")