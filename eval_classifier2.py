import numpy as np
from LogisticRegression import LogisticRegression

# Load test dataset
test_data = np.load('test_classification.npz')
X_test = test_data['X_test'][:, :2]  # Sepal Length & Sepal Width as input features
y_test = test_data['y_test']  # True class labels

# Load trained classifier
model = LogisticRegression()
model.load("model_classifier2.npz")

# Compute accuracy using the model's score method
accuracy = model.score(X_test, y_test)
print(f"Classification Accuracy on Test Data (Sepal Features): {accuracy:.4f}")