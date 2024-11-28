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

#### predict(self, inputs):
$$Z^{[1]}=W^{[1]}.X$$
$$A^{[1]}=ReLU(Z^{[1]})$$
$$Z^{[2]}=W^{[2]}A^{[1]}$$
$$A^{[2]}=\sigma(Z^{[2]})=\frac{1}{1+e^{-Z^{[2]}}}=Y_{pred}$$


#### w2 update:
$$W^{[2]} = W^{[2]} + \Delta W^{[2]}$$
$$\Delta W^{[2]} = - \alpha \frac{\partial cost}{\partial W^{[2]}}$$
$$\frac{\partial cost}{\partial W^{[2]}} = (\frac{-2}{n}(Y_{true}-A^{[2]})\odot A^{[2]}\odot (1-A^{[2]}))\bullet A^{[1]T}$$
$$W^{[2]}=W^{[2]}+(\frac{2 \alpha}{n}(Y_{true}-A^{[2]})\odot A^{[2]}\odot (1-A^{[2]}))\bullet A^{[1]T}$$

#### w1 update:
$$W^{[1]} = W^{[1]} + \Delta W^{[1]}$$
$$\Delta W^{[1]} = - \alpha \frac{\partial cost}{\partial W^{[1]}}$$

$$\frac{\partial cost}{\partial W^{[1]}} = (((\frac{-2}{n}(Y_{true}-A^{[2]})\odot A^{[2]}\odot (1-A^{[2]}))^T\bullet W^{[2]})^T\odot \frac{\partial A^{[1]}}{\partial Z^{[1]}}) \bullet X^T$$

$$W^{[1]}=W^{[1]}+(((\frac{2 \alpha}{n}(Y_{true}-A^{[2]})\odot A^{[2]}\odot (1-A^{[2]}))^T\bullet W^{[2]})^T\odot \frac{\partial A^{[1]}}{\partial Z^{[1]}}) \bullet X^T$$


```python
relu_gradient = np.where(A_1 > 0, 1, 0)
```
