import numpy as np
from MultiOutputLinearRegression import MultiOutputLinearRegression

# Evaluation script for Multi Output regression model

# Load test dataset
test_data = np.load('test_data.npz')
X_test = test_data['X_test'][:, :2]  # Use sepal length & width as input features
y_test = test_data['X_test'][:, 2:]  # Predict petal length & width

# Load trained multi-output regression model
model = MultiOutputLinearRegression()
model.load("model_multi_output.npz")

# Make predictions
y_pred = model.predict(X_test)

# Compute Mean Squared Error (MSE) across both outputs
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error on Test Data: {mse:.4f}")
