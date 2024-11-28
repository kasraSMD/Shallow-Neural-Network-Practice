# Shallow-Neural-Network-Practice
Shallow Neural Network for Pima Indians Diabetes Dataset

### Overview
This project implements a shallow neural network to predict diabetes using the well-known Pima Indians Diabetes dataset.

### Dataset
The Pima Indians Diabetes dataset contains 8 features related to health indicators (such as age, glucose level, BMI, etc.) and a binary target variable indicating whether the individual has diabetes.

### Steps Involved
1) Load and preprocess the dataset.
2) Design and implement a shallow neural network.
3) Train the model on the dataset.
4) Evaluate model

### The mathematical formulas used in this exercise are provided below:
#### Normalization :
$$ Z = \frac{x_i - \bar{x}}{\sigma} $$

```python
sigmoid_Z = 1 / (1 + np.exp(-Z))
```

```python
ReLU_Z = np.maximum(0, Z)
```

