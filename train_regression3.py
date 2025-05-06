import numpy as np
import matplotlib.pyplot as plt
from LinearRegression import LinearRegression

# Training model for single-output regression (Model 3)

# Load full dataset
data = np.load('train_data.npz')
X_train = data['X_train'][:, [0, 3]]  # Sepal Length & Petal Width as input
y_train = data['X_train'][:, 1]       # Sepal Width as target

# Train the model
model = LinearRegression(batch_size=32, max_epochs=100, patience=3)
model.fit(X_train, y_train)

# Save the model parameters
model.save("model_regression3.npz")

# Plot the training loss
plt.plot(model.train_loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve for Regression 3")
plt.legend()
plt.savefig("training_loss3.png")
plt.show()
