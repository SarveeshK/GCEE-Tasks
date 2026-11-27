
import numpy as np

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)


# Activation functions
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


# Reproducible initialization
np.random.seed(42)

# Network architecture: 2 -> 4 -> 1
W1 = np.random.randn(2, 4) * np.sqrt(2 / 2)
b1 = np.zeros((1, 4))

W2 = np.random.randn(4, 1) * np.sqrt(1 / 4)
b2 = np.zeros((1, 1))

learning_rate = 0.1
epochs = 10000
n = len(X)

for epoch in range(epochs):

    # -------------------------
    # 1. Forward propagation
    # -------------------------
    Z1 = X @ W1 + b1
    A1 = relu(Z1)

    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)

    # -------------------------
    # 2. Calculate loss
    # -------------------------
    eps = 1e-8

    loss = -np.mean(
        y * np.log(A2 + eps)
        + (1 - y) * np.log(1 - A2 + eps)
    )

    # -------------------------
    # 3. Backpropagation
    # -------------------------

    # Output layer gradient
    dZ2 = (A2 - y) / n

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    # Hidden layer gradient
    dA1 = dZ2 @ W2.T

    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # -------------------------
    # 4. Update weights and biases
    # -------------------------
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if (epoch + 1) % 1000 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss:.6f}"
        )


# -------------------------
# 5. Test the trained network
# -------------------------
Z1 = X @ W1 + b1
A1 = relu(Z1)

Z2 = A1 @ W2 + b2
predictions = sigmoid(Z2)

print("\nFinal Predictions:")

for inputs, prediction in zip(X, predictions):
    predicted_class = int(prediction[0] >= 0.5)

    print(
        f"Input: {inputs.astype(int)}, "
        f"Probability: {prediction[0]:.4f}, "
        f"Predicted class: {predicted_class}"
    )
