import numpy as np
import json

class LogisticRegression:
    def __init__(self, batch_size=32, learning_rate=0.01, max_epochs=100, patience=3, multi_class=False):
        """
        Logistic Regression Classifier with Batch Gradient Descent and Early Stopping.

        Parameters:
        -----------
        batch_size : int
            Number of samples per batch.
        learning_rate : float
            Step size for gradient descent.
        max_epochs : int
            Maximum number of training epochs.
        patience : int
            Number of epochs to wait for validation loss improvement before stopping.
        multi_class : bool
            If True, uses Softmax for multi-class classification. Otherwise, uses Sigmoid.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.patience = patience
        self.multi_class = multi_class  # Multi-class classification flag
        self.weights = None
        self.bias = None
        self.train_loss_history = []
        self.val_loss_history = []

    def sigmoid(self, z):
        """Compute the Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def softmax(self, z):
        """Compute the Softmax activation function for multi-class classification."""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Numerical stability
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, y_true, y_pred):
        """Compute loss: Binary Cross-Entropy for binary, Categorical Cross-Entropy for multi-class."""
        n_samples = y_true.shape[0]

        if self.multi_class:
            loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / n_samples
        else:
            loss = -np.mean(y_true * np.log(y_pred + 1e-9) + (1 - y_true) * np.log(1 - y_pred + 1e-9))

        return loss

    def fit(self, X, y):
        """
        Train the logistic regression model using batch gradient descent with early stopping.

        Parameters:
        -----------
        X : numpy.ndarray
            Training data (n_samples, n_features)
        y : numpy.ndarray
            Target labels (n_samples,) for binary or (n_samples, n_classes) for multi-class
        """
        n_samples, n_features = X.shape

        # Convert y to one-hot encoding if multi-class. 
        # Process of converting multiple classes (y-values) into vector of matrix.
        if self.multi_class:
            n_classes = len(np.unique(y))
            y_one_hot = np.zeros((n_samples, n_classes))
            y_one_hot[np.arange(n_samples), y] = 1
            y = y_one_hot # One-hot encoded matrix
            self.weights = np.random.randn(n_features, n_classes) * 0.01
            self.bias = np.zeros((1, n_classes)) # One bias of 0 for each class.
        else:
            self.weights = np.random.randn(n_features) * 0.01
            self.bias = 0.0

        # Split data into 90% training and 10% validation
        split_index = int(0.9 * n_samples)
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        best_loss = float('inf')
        no_improvement_count = 0
        best_weights = self.weights.copy()
        best_bias = self.bias

        for epoch in range(self.max_epochs):
            # Shuffle training data each epoch
            indices = np.arange(len(X_train))
            np.random.shuffle(indices)
            X_train_shuffled, y_train_shuffled = X_train[indices], y_train[indices]

            epoch_loss = 0

            # Batch training
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train_shuffled[i:i + self.batch_size]
                y_batch = y_train_shuffled[i:i + self.batch_size]

                # Compute predictions
                z = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.softmax(z) if self.multi_class else self.sigmoid(z)

                # Compute loss
                loss = self.compute_loss(y_batch, y_pred)
                epoch_loss += loss

                # Compute gradients
                error = y_pred - y_batch
                grad_weights = np.dot(X_batch.T, error) / len(X_batch)
                grad_bias = np.sum(error, axis=0) / len(X_batch)

                # Update weights
                self.weights -= self.learning_rate * grad_weights
                self.bias -= self.learning_rate * grad_bias

            # Store training loss per epoch
            avg_train_loss = epoch_loss / (len(X_train) / self.batch_size)
            self.train_loss_history.append(avg_train_loss)

            # Compute validation loss
            z_val = np.dot(X_val, self.weights) + self.bias
            y_val_pred = self.softmax(z_val) if self.multi_class else self.sigmoid(z_val)
            val_loss = self.compute_loss(y_val, y_val_pred)
            self.val_loss_history.append(val_loss)

            # Early stopping
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = self.weights.copy()
                best_bias = self.bias
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                if no_improvement_count >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Use best parameters
        self.weights = best_weights
        self.bias = best_bias

        print(f"Final Training Loss: {self.train_loss_history[-1]:.4f}")

    def predict(self, X):
        """
        Predict class labels using trained weights.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data (n_samples, n_features)

        Returns:
        -----------
        numpy.ndarray
            Predicted class labels (n_samples,)
        """
        # Compute the linear combination: z = XW + b
        z = np.dot(X, self.weights) + self.bias

        # Apply activation function (Sigmoid for binary, Softmax for multi-class)
        if self.multi_class:
            y_pred = self.softmax(z)
            return np.argmax(y_pred, axis=1)  # Return class with highest probability
        else:
            y_pred = self.sigmoid(z)
            return (y_pred >= 0.5).astype(int)  # Convert probabilities to binary classes. 1 or 0.

    def score(self, X, y):
        """
        Compute classification accuracy.

        Parameters:
        -----------
        X : numpy.ndarray
            Input data (n_samples, n_features)
        y : numpy.ndarray
            True class labels (n_samples,)

        Returns:
        -----------
        float
            Accuracy score
        """
        y_pred = self.predict(X) # Predict labels.
        return np.mean(y_pred == y) # Compare with actual lables and takes the mean of the correct predictions to measure the accuracy %.

    def save(self, file_path):
        """Save model parameters to a file."""
        model_params = {"weights": self.weights.tolist(), "bias": self.bias.tolist(), "multi_class": self.multi_class}
        with open(file_path, 'w') as f:
            json.dump(model_params, f)
        print(f"Model parameters saved to {file_path}")

    def load(self, file_path):
        """Load model parameters from a file."""
        with open(file_path, 'r') as f:
            model_params = json.load(f)
        self.weights = np.array(model_params["weights"])
        self.bias = np.array(model_params["bias"])
        self.multi_class = model_params["multi_class"]
        print(f"Model parameters loaded from {file_path}")
