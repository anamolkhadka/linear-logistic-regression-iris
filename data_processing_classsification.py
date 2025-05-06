import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def load_and_split_data():
    # Load the Iris dataset
    iris = load_iris()
    X = iris.data  # All 4 features (sepal length, sepal width, petal length, petal width)
    y = iris.target  # Class labels (0 = Setosa, 1 = Versicolor, 2 = Virginica)

    # Split dataset into training (90%) and testing (10%), ensuring an even class split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Save the processed data
    np.savez('train_classification.npz', X_train=X_train, y_train=y_train)
    np.savez('test_classification.npz', X_test=X_test, y_test=y_test)

if __name__ == "__main__":
    load_and_split_data()
    print("Classification data has been prepared and saved.")
