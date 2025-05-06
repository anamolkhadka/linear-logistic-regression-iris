import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# Training model for single-output regression (Model 1) with Regularization

# Load data
data = np.load('train_data.npz')
X_train = data['X_train'][:, :2]  # Selecting sepal length & width as input
y_train = data['X_train'][:, 2]   # Predicting petal length (3rd column)

# Train model with L2 Regularization
regularization_value = 0.1  # Try different values like 0.01, 0.1, 1.0, etc.
model_reg = LinearRegression(batch_size=32, max_epochs=100, patience=3, regularization=regularization_value)
model_reg.fit(X_train, y_train)

# Save regularized model
model_reg.save("model_regression1_reg.npz")

# Plot training loss
plt.plot(model_reg.train_loss_history, label="Training Loss (Regularized)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve for Regression 1 (Regularized)")
plt.legend()
plt.savefig("training_loss1_reg.png")
plt.show()
