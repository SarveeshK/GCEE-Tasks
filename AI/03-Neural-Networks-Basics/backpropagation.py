
# Training data
x = 2.0
y = 3.0

# Initial parameters
weight = 0.5
bias = 0.0

# Learning rate
learning_rate = 0.1

for epoch in range(10):

    # 1. Forward propagation
    prediction = weight * x + bias

    # 2. Calculate loss
    loss = 0.5 * (prediction - y) ** 2

    # 3. Backpropagation
    error = prediction - y

    weight_gradient = error * x
    bias_gradient = error

    # 4. Update parameters
    weight -= learning_rate * weight_gradient
    bias -= learning_rate * bias_gradient

    print(
        f"Epoch: {epoch + 1}, "
        f"Prediction: {prediction:.4f}, "
        f"Loss: {loss:.6f}"
    )

print("\nFinal Weight:", weight)
print("Final Bias:", bias)
print("Final Prediction:", weight * x + bias)
