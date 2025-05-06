import numpy as np
import matplotlib.pyplot as plt
from MultiOutputLinearRegression import MultiOutputLinearRegression

# Training script for the Multi Output Linear Regression
# Load data
data = np.load('train_data.npz')
X_train = data['X_train'][:, :2]  # Sepal length & width
y_train = data['X_train'][:, 2:]  # Petal length & width

# Train model
model = MultiOutputLinearRegression()
model.fit(X_train, y_train)

# Save model
model.save("model_multi_output.npz")

# Plot training loss
plt.plot(model.train_loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve for Multi-Output Regression")
plt.legend()
plt.savefig("training_loss_multi.png")
plt.show()