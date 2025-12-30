# Neural Network Fundamentals - Perceptron

Understand foundational neural network concepts by implementing the perceptron algorithm from scratch.

## 📋 Project Information

- **Notebook:** `Neural Network - Perceptron Algorithm.ipynb`
- **Topic:** Neural Networks - Perceptron Basics
- **Dataset:** Synthetic Binary Classification
- **Algorithm:** Perceptron from scratch
- **Complexity:** Intermediate

## 🎯 Learning Objectives

In this project, you will learn:
- Memahami konsep dasar neural networks
- Implementasi Perceptron algorithm from scratch
- Fungsi aktivasi dan decision boundaries
- Forward propagation
- Learning process and weight updates
- Limitations of a single perceptron

## 📋 Key Topics

### 1. Perceptron Architecture
The perceptron is the most basic computational unit in neural networks:
- **Input layer:** Receives input features
- **Weights:** Parameters to be learned
- **Bias:** Additional learnable parameter
- **Activation function:** Non-linearity (step, sigmoid, ReLU)
- **Output:** Binary prediction (0 or 1)

### 2. Mathematical Model

**Perceptron Formula:**
$$z = w_1x_1 + w_2x_2 + ... + w_nx_n + b$$

$$y = \text{activation}(z)$$

**Step Function (Binary):**
$$f(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$$

**Sigmoid Function:**
$$f(z) = \frac{1}{1 + e^{-z}}$$

### 3. Learning Rule

**Perceptron Learning Rule:**
- Jika prediksi salah: Update weights
- $$w_{new} = w_{old} + \eta \times (y_{true} - y_{pred}) \times x$$
- $$b_{new} = b_{old} + \eta \times (y_{true} - y_{pred})$$
- Dimana $\eta$ adalah learning rate

## 🛠️ Requirements

```bash
pip install numpy matplotlib scikit-learn
```

## 📚 Library yang Digunakan

| Library | Fungsi |
|---------|--------|
| **numpy** | Numerical operations |
| **matplotlib** | Visualization & plotting |
| **scikit-learn** | Dataset generation, metrics |

## 🚀 Cara Menjalankan

### Local Machine
```bash
jupyter notebook Neural\ Network\ -\ Perceptron\ Algorithm.ipynb
```

### Google Colab
1. Upload notebook ke Colab
2. Run cells dari atas ke bawah
3. Interact dengan visualizations

## 📝 Inti Notebook

### 1. **Import Library**
Loading dependencies

### 2. **Dataset Generation/Loading**
- Generate synthetic binary classification data
- Atau load built-in dataset (e.g., Iris, make_classification)
- Visualisasi dataset

### 3. **Perceptron Implementation from Scratch**

```python
class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        # Initialize weights & bias
        self.weights = np.zeros((X.shape[1], 1))
        self.bias = 0
        
        # Training loop
        for epoch in range(self.epochs):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.step_function(z)
            
            # Calculate errors
            errors = y - y_pred
            
            # Update weights & bias
            self.weights += self.lr * np.dot(X.T, errors)
            self.bias += self.lr * np.sum(errors)
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.step_function(z)
    
    def step_function(self, z):
        return np.where(z > 0, 1, 0)
```

### 4. **Training Process**
- Train perceptron dengan training data
- Monitor error/loss per epoch
- Visualisasi learning curve

### 5. **Decision Boundary Visualization**
- Plot data points dengan colors per class
- Plot decision boundary (garis/hyperplane)
- Visualisasi bagaimana perceptron membagi space

### 6. **Model Evaluation**
- Accuracy pada training data
- Accuracy pada test data (jika ada)
- Confusion matrix
- Classification report

### 7. **Learning Rate Impact**
- Train dengan berbagai learning rates
- Compare convergence speed & final accuracy
- Visualisasi effect

### 8. **Activation Function Comparison**
- Step function (original)
- Sigmoid function (smooth)
- ReLU (modern)
- Compare behavior & outputs

### 9. **Limitations Discussion**
- XOR problem (tidak solvable dengan single perceptron)
- Linear separability requirement
- Need untuk multi-layer networks
- Visualisasi XOR case yang gagal

### 10. **Extension to Multi-layer**
- Introduce neural network dengan hidden layers
- Explain bagaimana multi-layer overcome limitations
- Teaser untuk backpropagation

## 📈 Output yang Dihasilkan

- **Learning Curves** - Error vs epochs
- **Decision Boundaries** - Visualization of learned boundaries
- **Accuracy Metrics** - Training & test performance
- **Weight Visualizations** - Final learned weights
- **Comparative Plots** - Different learning rates/activations
- **XOR Problem Illustration** - Why single perceptron fails

## 💡 Key Concepts

### Perceptron
Simplest form dari artificial neuron:
- Binary classifier
- Linear decision boundary
- Works untuk linearly separable data

### Linear Separability
Data dimana single line/hyperplane dapat separate dua classes

### Decision Boundary
Hyperplane yang memisahkan two classes dalam feature space

### Learning Rate (η)
Hyperparameter yang control step size dalam weight updates:
- Too small: Slow learning
- Too large: Overshoot, diverge

### XOR Problem
Classic problem yang menunjukkan limitation perceptron:
```
XOR Truth Table:
x1  x2  y
0   0   0
0   1   1
1   0   1
1   1   0
```
- Tidak linearly separable
- Single perceptron tidak bisa solve
- Need minimal 2 layers

### Activation Functions

**Step Function:**
- Binary output (0 atau 1)
- Non-differentiable
- Original perceptron

**Sigmoid:**
- Smooth output (0 to 1)
- Differentiable
- Used dalam hidden/output layers

**ReLU (Rectified Linear Unit):**
- Modern choice
- $f(z) = \max(0, z)$
- Computationally efficient

## ⚠️ Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Weights don't converge | Check learning rate, is data linearly separable? |
| All predictions same class | Bad initialization, try different LR |
| Slow convergence | Increase learning rate (carefully) |
| Oscillating error | Learning rate too large |

## 💻 Key Code Patterns

```python
# Forward pass
z = np.dot(X, w) + b

# Activation
y_pred = sigmoid(z)  # or step(z)

# Calculate loss
error = y_true - y_pred

# Update weights
dw = np.dot(X.T, error) / m
w = w + learning_rate * dw
```

## 📚 Referensi

- [Perceptron Algorithm - Wikipedia](https://en.wikipedia.org/wiki/Perceptron)
- [Neural Networks Basics](https://en.wikipedia.org/wiki/Artificial_neural_network)
- [The XOR Problem](https://en.wikipedia.org/wiki/XOR)
- [Activation Functions](https://en.wikipedia.org/wiki/Activation_function)
- [Learning Rate in Neural Networks](https://machinelearningmastery.com/learning-rate-for-deep-learning/)

## ✅ Checklist

- [ ] Understand perceptron concept
- [ ] Implement perceptron from scratch
- [ ] Generate/load dataset
- [ ] Train perceptron
- [ ] Visualize decision boundary
- [ ] Evaluate performance
- [ ] Experiment dengan learning rates
- [ ] Test activation functions
- [ ] XOR problem analysis
- [ ] Understand limitations

---

**Author:** Abdi Wijaya Sasmita (4222201044)  
**Date:** December 2025  
**Status:** ✓ Complete

## 🔗 Next Steps

Setelah memahami perceptron, lanjutkan ke:
1. **Multi-layer Neural Networks (MLPs)**
2. **Backpropagation Algorithm**
3. **Deep Learning dengan TensorFlow/Keras**
4. **Convolutional Neural Networks (CNN)** untuk image
5. **Recurrent Neural Networks (RNN)** untuk sequence
