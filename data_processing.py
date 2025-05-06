import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_and_split_data():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # All 4 features. Sepal length, Sepal width, petal length, petal width

    # Split dataset into training (90%) and testing (10%)
    X_train, X_test = train_test_split(X, test_size=0.1, random_state=42)

    # Save full train and test sets **without pre-selecting X and y**
    np.savez('train_data.npz', X_train=X_train)
    np.savez('test_data.npz', X_test=X_test)

if __name__ == "__main__":
    load_and_split_data()
    print("Data has been prepared and saved.")
