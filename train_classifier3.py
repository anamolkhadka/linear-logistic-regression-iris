import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression

# Training Model for Classifier 3. Using all 4 features.

# Load data
data = np.load('train_classification.npz')
X_train = data['X_train']  # All 4 features
y_train = data['y_train']  # Class labels

# Train Logistic Regression Model
model = LogisticRegression(learning_rate=0.01, max_epochs=1000, multi_class=True)
model.fit(X_train, y_train)

# Save model
model.save("model_classifier3.npz")

# Plot training loss
plt.plot(model.train_loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve for Classifier 3 (All Features)")
plt.legend()
plt.savefig("training_loss_classifier3.png")
plt.show()