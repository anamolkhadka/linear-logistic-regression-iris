import numpy as np
from LinearRegression import LinearRegression

# Evaluation script for single-output regression (Model 1)

# Load test dataset
test_data = np.load('test_data.npz')
X_test = test_data['X_test'][:, :2]  # Use same input features as training (sepal length & width)
y_test = test_data['X_test'][:, 2]   # Predicting petal length (same as in training)

# Load trained model
model = LinearRegression()
model.load("model_regression1.npz")

# Make predictions
y_pred = model.predict(X_test)

# Compute Mean Squared Error (MSE)
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error on Test Data: {mse:.4f}")