
import numpy as np
import matplotlib.pyplot as plt


# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


# Input values
x = np.linspace(-5, 5, 200)

# Plot activation functions
plt.figure(figsize=(10, 6))

plt.plot(x, sigmoid(x), label="Sigmoid")
plt.plot(x, tanh(x), label="Tanh")
plt.plot(x, relu(x), label="ReLU")
plt.plot(x, leaky_relu(x), label="Leaky ReLU")

plt.title("Comparison of Activation Functions")
plt.xlabel("Input")
plt.ylabel("Output")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Softmax example for multiclass classification
scores = np.array([2.0, 1.0, 0.1])
probabilities = softmax(scores)

print("Input scores:", scores)
print("Softmax probabilities:", probabilities)
print("Probability sum:", probabilities.sum())
