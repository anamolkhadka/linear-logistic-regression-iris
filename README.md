# 🌸 Iris Dataset: Linear and Logistic Regression from Scratch

This project explores regression and classification techniques applied to the Iris dataset using models built **entirely from scratch** in Python. The goal is to understand how Linear Regression and Logistic Regression work under the hood, including model training, optimization, and evaluation without using high-level ML libraries for modeling.

---

## 🔍 Overview

This project includes:

- 🧮 **Linear Regression**
  - Single-output and Multi-output models
  - Batch gradient descent optimization
  - Early stopping with validation set
  - L2 Regularization support
  - Mean Squared Error (MSE) evaluation

- 🧠 **Logistic Regression**
  - Multi-class classification using Softmax
  - Gradient descent optimization
  - Accuracy evaluation
  - Decision boundary visualization

All models are trained and evaluated using the classic **Iris dataset**.

---

## 📈 Linear Regression

### ✔️ Features

- Implements Linear Regression from scratch using NumPy
- Supports:
  - Batch gradient descent
  - Early stopping
  - L2 regularization
- Saves model weights and loss plots
- Evaluation via Mean Squared Error (MSE)

### 💡 Experiments

| Model | Input Features                 | Target Output  | MSE     |
|-------|--------------------------------|----------------|---------|
| 1     | Sepal Length                   | Sepal Width    | 0.0791  |
| 2     | Petal Length                   | Petal Width    | 0.0608  |
| 3     | Sepal Width + Petal Width      | Sepal Length   | 0.0724  |
| 4     | Sepal Width + Petal Length     | Petal Width    | **0.0538** ✅ Best

### 🔁 Regularization

Model 1 was retrained using L2 regularization. The new weights were compared against the non-regularized model to observe overfitting control and parameter shrinkage.

### 📐 Multi-output Regression

Predicted both **Petal Length and Petal Width** from **Sepal Length and Width** using an extended model:
- Evaluated using extended MSE across output dimensions.
- Training losses and final MSE were plotted and saved.

---

## 🧠 Logistic Regression (Classification)

### ✔️ Features

- Implements multi-class Logistic Regression from scratch
- Uses Softmax for multi-class output
- Trained using gradient descent
- Decision region visualization using `mlxtend`

### 💡 Experiments

| Classifier | Features Used            | Accuracy |
|------------|--------------------------|----------|
| 1          | Petal Length & Width     | 33.33%   |
| 2          | Sepal Length & Width     | 33.33%   |
| 3          | All 4 features           | **93.33%** ✅ Best

### 📌 Observations

- Classifier 3 significantly outperforms others due to better feature coverage.
- Decision boundaries were visualized for Classifier 1 and 2 to highlight poor separation using limited features.

---

## 📊 Visualizations & Outputs

- `training_loss_*.png`: Training loss plots for regression and classification
- `decision_boundary_classifierX.png`: Decision regions for classifiers using 2D inputs
- `.npz` files: Saved model parameters and datasets

---

## 🛠 How to Run

Make sure to process the dataset first before training:

```bash
# For regression
python data_processing.py

# For classification
python data_processing_classification.py
```

## 🎓 Key Learnings

- Implementing ML algorithms manually helps demystify optimization and evaluation processes.
- Regularization plays a crucial role in reducing overfitting, especially for small datasets.
- Visualization of decision boundaries and loss curves aids in model interpretability.
- Feature selection and dimensionality have a large impact on model performance.

## 📜 License

- #### © 2025 Anamol Khadka. All rights reserved.
- This work is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.
- You are free to **share** and **adapt** the material for any purpose, even commercially, **as long as appropriate credit is given**.
- Unauthorized reproduction, distribution, or modification of this work **without attribution** is prohibited.
- For permissions, inquiries, or attribution guidelines, please contact: **khadkaanamol8@gmail.com**

🔗 [View License Details](https://creativecommons.org/licenses/by/4.0/)
