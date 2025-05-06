import numpy as np
import json

# Implementation of multi-output Linear regression class.

class MultiOutputLinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression for Multiple Outputs using Gradient Descent."""
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None  # Shape: (n_features, n_outputs)
        self.bias = None  # Shape: (1, n_outputs)
        self.train_loss_history = [] # To store training loss values
        self.val_loss_history = [] # To store validation loss values

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Train the multi-output linear regression model."""
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # Get the number of samples, features, and output dimensions
        n_samples, n_features = X.shape # Changed for MLR
        _, n_outputs = y.shape # Changed for MLR

        # Initialize weights (random small values) and bias (zeros)
        self.weights = np.random.randn(n_features, n_outputs) * 0.01 # Changed for MLR. Shape: (d. m)
        self.bias = np.zeros((1, n_outputs)) # Shape (m, )
        learning_rate = 0.01  # Learning rate

        # Split dataset into 90% training and 10% validation
        split_index = int(0.9 * n_samples)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        best_loss = float('inf')
        no_improvement_count = 0
        best_weights = self.weights.copy()
        best_bias = self.bias.copy()

        # Training loop
        for epoch in range(max_epochs):
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

            epoch_loss = 0

            # Batch processing
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                # Compute predictions: y_pred = XW + b
                y_pred = np.dot(X_batch, self.weights) + self.bias # Shape: (n, m)

                # Compute gradient
                error = y_pred - y_batch  # Shape: (n, m)
                grad_weights = (2 / len(X_batch)) * np.dot(X_batch.T, error) + self.regularization * self.weights  # Shape: (d, m)
                grad_bias = (2 / len(X_batch)) * np.mean(error, axis=0)  # Shape: (m,)


                # Update weights
                self.weights -= learning_rate * grad_weights
                self.bias -= learning_rate * grad_bias

                # Compute batch loss and accumulate for the epoch
                batch_loss = np.mean((y_pred - y_batch) ** 2)
                epoch_loss += batch_loss

            # Compute average loss for this epoch
            avg_train_loss = epoch_loss / (len(X_train) / batch_size)
            self.train_loss_history.append(avg_train_loss)

            # Compute validation loss
            val_pred = np.dot(X_val, self.weights) + self.bias
            val_loss = np.mean((val_pred - y_val) ** 2)
            self.val_loss_history.append(val_loss)

            # Early stopping logic
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = self.weights.copy()
                best_bias = self.bias.copy()
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Use the best model parameters after training
        self.weights = best_weights
        self.bias = best_bias

    def predict(self, X):
        """Predict using the multi-output linear model."""
        if self.weights is None or self.bias is None:
            raise ValueError("Model is not trained yet. Call `fit()` first.")
        return np.dot(X, self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the model using Mean Squared Error."""
        y_pred = self.predict(X)
        squared_errors = (y - y_pred) ** 2
        n_samples, n_outputs = y.shape
        mse = np.sum(squared_errors) / (n_samples * n_outputs) # MSE loss function changed for MLR
        return mse

    def save(self, file_path):
        """Save model parameters to a file."""
        if self.weights is None or self.bias is None:
            raise ValueError("Model is not trained yet. Train first before saving.")
        model_params = {
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist()
        }
        with open(file_path, 'w') as f:
            json.dump(model_params, f)
        print(f"Model parameters saved to {file_path}")

    def load(self, file_path):
        """Load model parameters from a file."""
        with open(file_path, 'r') as f:
            model_params = json.load(f)
        self.weights = np.array(model_params["weights"])
        self.bias = np.array(model_params["bias"])
        print(f"Model parameters loaded from {file_path}")
