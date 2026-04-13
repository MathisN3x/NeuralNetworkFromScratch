# 🧠 Neural Network From Scratch

This project implements a simple feedforward neural network written from scratch in Python using **NumPy**. It is a pedagogical implementation designed to bridge the gap between mathematical theory and computational practice.

---

## 🎯 Project Goals
- **Full implementation:** Build a deep neural network from the ground up.
- **Mathematical Mastery:**
  - Weight and Bias initialization.
  - Forward propagation using matrix vectorization.
  - Backpropagation (Chain Rule application).
  - Parameter optimization via Gradient Descent.
- **Visual Intuition:** Observe how the decision boundary adapts to data in real-time.

---

## 🔢 Mathematical Overview

### Notation
For a network with $L$ layers and $m$ examples:

- $X \in \mathbb{R}^{n^{[0]} \times m}$ (Input matrix)
- $Y \in \mathbb{R}^{n^{[L]} \times m}$ (Ground truth)

For each layer $l$:
- $W^{[l]}$ : Weights matrix
- $b^{[l]}$ : Bias vector
- $Z^{[l]} = W^{[l]} A^{[l-1]} + b^{[l]}$
- $A^{[l]} = g^{[l]}(Z^{[l]})$

---

### Activation Functions

| Function | Equation | Derivative |
| :--- | :--- | :--- |
| **Sigmoid** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ |
| **ReLU** | $ReLU(z) = max(0, z)$ | $1$ if $z > 0$, else $0$ |

---

### Loss Function (Binary Cross-Entropy)

$$J = -\frac{1}{m} \sum_{i=1}^{m} [ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) ]$$

---

### Backpropagation Equations

Vectorized gradients for layer $l$:

**1. Output Layer Error:**
$$dZ^{[L]} = A^{[L]} - Y$$

**2. Hidden Layer Error:**
$$dZ^{[l]} = (W^{[l+1]})^T dZ^{[l+1]} \cdot g'^{[l]}(Z^{[l]})$$

**3. Gradients:**
$$dW^{[l]} = \frac{1}{m} dZ^{[l]} (A^{[l-1]})^T$$
$$db^{[l]} = \frac{1}{m} \sum dZ^{[l]}$$

---

## 🧩 Training Loop (Pseudocode)

```python
# Initialize weights and biases
parameters = initialize_parameters(layer_dims)

for epoch in range(epochs):
    # 1. Forward Pass
    AL, caches = forward_propagation(X, parameters)
    
    # 2. Compute Loss
    cost = compute_cost(AL, Y)
    
    # 3. Backward Pass (Gradients)
    grads = backward_propagation(AL, Y, caches)
    
    # 4. Update Weights
    parameters = update_parameters(parameters, grads, learning_rate)
