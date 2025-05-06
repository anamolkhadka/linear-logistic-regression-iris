import numpy as np
import matplotlib.pyplot as plt
from LogisticRegression import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.preprocessing import StandardScaler

# Training Model for Classifier 1 (Petal Length & Petal Width)

# Load dataset
data = np.load('train_classification.npz')
X_train = data['X_train'][:, 2:]  # Selecting Petal Length & Petal Width as features
y_train = data['y_train']  # Class labels

# Standardize the features (important for good decision boundaries)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train Logistic Regression Model
model = LogisticRegression(learning_rate=0.01, max_epochs=1000, multi_class=True)
model.fit(X_train, y_train)

# Save model parameters
model.save("model_classifier1.npz")

# Plot training loss curve
plt.plot(model.train_loss_history, label="Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve for Classifier 1")
plt.legend()
plt.savefig("training_loss_classifier1.png")
plt.show()

# Visualize Decision Boundaries
plt.figure(figsize=(8, 6))
plot_decision_regions(X_train, y_train, clf=model)
plt.title("Decision Boundaries for Classifier 1 (Petal Features)")
plt.savefig("decision_boundary_classifier1.png")
plt.show()