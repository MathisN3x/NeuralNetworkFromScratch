# 🧠 Neural Network From Scratch

This project implements a simple feedforward neural network written from scratch in Python using NumPy (no TensorFlow / PyTorch). It's intended to be a pedagogical implementation to help you understand the mathematics and basic mechanics behind neural networks, visualized in 2D.

It lets you:
- Understand the core steps of a neural network
- Visualize model training and the decision boundary in 2D
- Observe how the decision boundary evolves during training

---

## 🎯 Project goals

- Implement a neural network from scratch
- Learn to:
  - Initialize weights and biases
  - Perform forward propagation
  - Compute gradients using backpropagation
  - Update parameters with gradient descent
- Visualize the evolving decision boundary

---
## 🔢 Mathematical overview

### Notation

For a network with \( L \) layers and \( m \) examples:

$$
\mathbf{X} \in \mathbb{R}^{n^{[0]} \times m}, \quad
\mathbf{Y} \in \mathbb{R}^{n^{[L]} \times m}
$$

For layer \( l \):

$$
\mathbf{W}^{[l]} \in \mathbb{R}^{n^{[l]} \times n^{[l-1]}}
$$

$$
\mathbf{b}^{[l]} \in \mathbb{R}^{n^{[l]} \times 1}
$$

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})
$$

---

### Forward propagation

$$
\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}
$$

$$
\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})
$$

---

### Activation functions

Sigmoid:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
\sigma'(z) = \sigma(z)(1 - \sigma(z))
$$

ReLU:
$$
\text{ReLU}(z) = \max(0, z)
$$

Derivative of ReLU:
- equals 1 if \( z > 0 \)
- equals 0 otherwise

Softmax:
$$
\text{softmax}(z)_i =
\frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

---

### Loss functions

Binary cross-entropy:
$$
J = -\frac{1}{m} \sum_{i=1}^{m}
\left[
y^{(i)} \log a^{(i)} +
(1 - y^{(i)}) \log(1 - a^{(i)})
\right]
$$

Categorical cross-entropy:
$$
J = -\frac{1}{m}
\sum_{i=1}^{m}
\sum_{k=1}^{K}
y_k^{(i)} \log a_k^{(i)}
$$

---

### Backpropagation

Output layer:
$$
\mathbf{dz}^{[L]} = \mathbf{a}^{[L]} - \mathbf{y}
$$

$$
\mathbf{dW}^{[L]} =
\frac{1}{m} \mathbf{dz}^{[L]} (\mathbf{a}^{[L-1]})^T
$$

$$
\mathbf{db}^{[L]} =
\frac{1}{m} \sum_{i=1}^{m} \mathbf{dz}^{[L]}_{:,i}
$$

Hidden layer:
$$
\mathbf{dz}^{[l]} =
(\mathbf{W}^{[l+1]})^T \mathbf{dz}^{[l+1]}
\odot g'(\mathbf{z}^{[l]})
$$

$$
\mathbf{dW}^{[l]} =
\frac{1}{m} \mathbf{dz}^{[l]} (\mathbf{a}^{[l-1]})^T
$$

$$
\mathbf{db}^{[l]} =
\frac{1}{m} \sum_{i=1}^{m} \mathbf{dz}^{[l]}_{:,i}
$$

---

### Gradient descent

$$
\mathbf{W}^{[l]} =
\mathbf{W}^{[l]} - \alpha \mathbf{dW}^{[l]}
$$

$$
\mathbf{b}^{[l]} =
\mathbf{b}^{[l]} - \alpha \mathbf{db}^{[l]}
$$

## 🧩 High-level training loop (pseudocode)

```python
# X: input matrix (n_x, m)
# Y: labels (n_y, m)
initialize_parameters()
for epoch in range(epochs):
    # Forward pass
    caches = forward_propagation(X, parameters)
    # Compute loss
    loss = compute_loss(caches["A_L"], Y)
    # Backward pass
    grads = backward_propagation(caches, Y)
    # Update parameters
    parameters = update_parameters(parameters, grads, learning_rate)
    # Optionally: record/plot decision boundary and loss
```

---

## 📷 Visualization

The repository includes a demonstration image (demo.png) that animates the model's decision boundary evolving during training over a 2D dataset (scikit-learn make_blobs). The visualization helps build intuition for how model parameters change and how the boundary adapts to the data.

---

## 🛠️ Technologies used

- Python — main language
- NumPy — matrix computation
- Matplotlib — plotting and visualization
- Scikit-learn — dataset generation (make_blobs)
- IPython.display — dynamic animation (in the notebook)

---

## 📁 Project structure

```bash
📦 NeuralNetworkFromScratch/
├── neural_network.ipynb    # Main Jupyter notebook with implementation + visualizations
├── demo.png                # Demo animation of the decision boundary
└── README.md               # This file
```

---

## 🚀 How to run

1. Install dependencies (recommended in a virtualenv):
   - numpy, matplotlib, scikit-learn, jupyter
2. Open `neural_network.ipynb` in Jupyter Notebook / JupyterLab
3. Run cells to see step-by-step implementation and the animated decision boundary

---

If you'd like, I can:
- Add more detailed derivations for backpropagation step-by-step,
- Add equations inline inside the notebook cells,
- Or convert the training pseudocode into a concrete Python module with unit tests.
