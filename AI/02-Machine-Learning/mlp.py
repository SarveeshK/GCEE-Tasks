
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# XOR dataset
X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [0, 1, 1, 0]

# Create the model
model = MLPClassifier(
    hidden_layer_sizes=(4,),
    activation="relu",
    solver="lbfgs",
    max_iter=5000,
    random_state=42
)

# Train the model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

print("Predictions:", predictions)
print("Actual:", y)

print(
    "Accuracy:",
    accuracy_score(y, predictions)
)
