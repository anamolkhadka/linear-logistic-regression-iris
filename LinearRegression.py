import numpy as np
import json

# Implementation of single-output Linear Regression

class LinearRegression:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent."""
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.train_loss_history = [] # To store training loss values
        self.val_loss_history = []    # To store validation loss values

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model using gradient descent with early stopping and loss tracking."""

        # Store parameters.
        self.batch_size = batch_size # The number of samples per batch.
        self.regularization = regularization
        self.max_epochs = max_epochs # Epoch is one complete pass through the entire training dataset during the training process of a model.
        self.patience = patience # The number of epochs to wait before stopping if the validation loss

        # TODO: Initialize the weights and bias based on the shape of X and y.
        n_samples, n_features = X.shape # number of samples (rows) and number of features (columns)
        self.weights = np.random.randn(n_features) * 0.01 # Small random values
        self.bias = 0.0
        learning_rate = 0.01  # Example learning rate. Alpha in the update equation.

        # Split data into training (90%) and validation (10%)
        split_index = int(0.9 * n_samples)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        best_loss = float('inf')
        no_improvement_count = 0
        best_weights = self.weights.copy()
        best_bias = self.bias

        # TODO: Implement the training loop.
        for epoch in range(max_epochs):
            # Shuffle training data for each epoch
            indices = np.arange(len(X_train)) # Creates an index array in range.
            np.random.shuffle(indices) # Shuffle the indices array.
            X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices] # This will shuffle rows of the data sets using indices array. Advance indexing used.

            epoch_loss = 0  # To accumulate batch losses for training

            # Batch processing.
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                # Compute Predictions. y^ = X.w + b
                y_pred = np.dot(X_batch, self.weights) + self.bias

                # Compute gradients from MSE loss equation. (Mean Squared Error loss)
                error = y_pred - y_batch
                grad_weights = (2 / len(X_batch)) * np.dot(X_batch.T, error) + regularization * self.weights # Gradient for weights. 2/n * X^T(y^ - y) + λw
                grad_bias = (2 / len(X_batch)) * np.sum(error) # Gradient for bias. 2/n * sum of (y^ - y)

                # Update parameters. W(new) = w(old) - a * W'(x).
                self.weights -= learning_rate * grad_weights
                self.bias -= learning_rate * grad_bias

                # Calculate training loss for the current batch and accumulate
                batch_loss = np.mean((y_pred - y_batch) ** 2)
                epoch_loss += batch_loss
            

            # Store average training loss per epoch
            avg_train_loss = epoch_loss / (len(X_train) / batch_size) # Average of the all the batch errors.
            self.train_loss_history.append(avg_train_loss)
            
            # Calculate validation loss
            val_pred = np.dot(X_val, self.weights) + self.bias
            val_loss = np.mean((val_pred - y_val) ** 2)
            self.val_loss_history.append(val_loss)

            # Early stopping logic
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = self.weights.copy()
                best_bias = self.bias
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= patience: # If 3 consecutive steps where validation loss increases.
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Use the best parameters after training
        self.weights = best_weights
        self.bias = best_bias

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data. (n_samples * n_features)
        
        Returns
        ----------
        numpy.ndarray
            The predicted values (n_samples * 1)
        """
        # TODO: Implement the prediction function.

        # First check if the weights and bias are initialized from the training.
        if self.weights is None or self.bias is None:
            raise ValueError("Model weights and bias are not initialized. Pleae call the fit method first.")
        
        # Compute the predictions using the linear regression formula: y = XW + b
        y_pred = np.dot(X, self.weights) + self.bias

        return y_pred

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        
        Returns
        ----------
        float: The mean squared error.
        """
        # TODO: Implement the scoring function.

        # Step 1: Predict score using the predict method
        y_pred = self.predict(X)

        # Step 2: Calculate the squared differences
        squared_errors = (y - y_pred) ** 2

        # Step 3: Calculate MSE with both sample size (n) and output size (m)
        n_samples, m_outputs = y.shape  # Get number of samples and outputs
        mse = np.sum(squared_errors) / (n_samples * m_outputs)

        return mse

    def save(self, file_path):
        # Save the model parameters to a file.
        if self.weights is None or self.bias is None:
            raise ValueError("Model parameters are not initialized. Train the model first.")

        # Convert numpy arrays to lists for JSON serialization
        model_params = {
            "weights": self.weights.tolist(),
            "bias": self.bias
        }

        with open(file_path, 'w') as f:
            json.dump(model_params, f)
        print(f"Model parameters saved to {file_path}")
    
    def load(self, file_path):
        # Load the model parameters from a file.

        with open(file_path, 'r') as f:
            model_params = json.load(f)

        # Convert lists back to numpy arrays
        self.weights = np.array(model_params["weights"])
        self.bias = model_params["bias"]

        print(f"Model parameters loaded from {file_path}")
    
