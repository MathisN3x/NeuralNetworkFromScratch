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

This section summarizes the key equations used in a basic fully-connected feedforward network.

Notation:
- For layer l:
  - W^{[l]} : weight matrix of shape (n^{[l]}, n^{[l-1]})
  - b^{[l]} : bias vector of shape (n^{[l]}, 1)
  - z^{[l]} : pre-activation (W^{[l]} a^{[l-1]} + b^{[l]})
  - a^{[l]} : activation output of layer l
- For m examples, X ∈ R^{n^{[0]}×m}, Y ∈ R^{n^{[L]}×m}

Forward propagation (single layer):
- z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}
- a^{[l]} = g^{[l]}(z^{[l]})

Common activation functions:
- Sigmoid:
  - σ(z) = 1 / (1 + e^{-z})
  - σ'(z) = σ(z) (1 − σ(z))
- ReLU:
  - ReLU(z) = max(0, z)
  - derivative: 1_{z>0}
- Softmax (for multi-class outputs):
  - softmax(z)_i = exp(z_i) / sum_j exp(z_j)

Loss functions:
- Binary cross-entropy (for one output unit):
  - For one example: L(a, y) = −[ y log a + (1−y) log(1−a) ]
  - For m examples: J = −(1/m) ∑_{i=1}^m [ y^{(i)} log a^{(i)} + (1−y^{(i)}) log(1−a^{(i)}) ]
- Categorical cross-entropy (for K classes, softmax outputs):
  - J = −(1/m) ∑_{i=1}^m ∑_{k=1}^K y_k^{(i)} log a_k^{(i)}

Backpropagation (gradient recipes):

Output layer (example: sigmoid + binary cross-entropy):
- dz^{[L]} = a^{[L]} − y
- dW^{[L]} = (1/m) dz^{[L]} (a^{[L-1]})^T
- db^{[L]} = (1/m) ∑_{i=1}^m dz^{[L]}_{:,i}

Hidden layer l (with elementwise activation g):
- dz^{[l]} = (W^{[l+1]})^T dz^{[l+1]} * g'(z^{[l]})
- dW^{[l]} = (1/m) dz^{[l]} (a^{[l-1]})^T
- db^{[l]} = (1/m) ∑_{i=1}^m dz^{[l]}_{:,i}

Gradient descent parameter update:
- W^{[l]} := W^{[l]} − α dW^{[l]}
- b^{[l]} := b^{[l]} − α db^{[l]}

---

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
