import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Modify sigmoid_derivative to handle overflow
def sigmoid_derivative(x):
    # Clip values to avoid overflow
    x = np.clip(x, 1e-5, 1 - 1e-5)  # Ensures values are within a safe range
    return x * (1 - x)


# Example input (flattened image vectors)
X = np.array([[0.2, 0.4, 0.6, 0.8],  # Dog image example (simplified)
              [0.3, 0.5, 0.7, 0.9]])  # Cat image example (simplified)

# Example output (1 for dog, 0 for cat)
y = np.array([[1], [0]])

# Network architecture
input_layer_size = 4  # Number of features
hidden_layer1_size = 6  # First hidden layer neurons
hidden_layer2_size = 5  # Second hidden layer neurons
hidden_layer3_size = 4  # Third hidden layer neurons
output_layer_size = 1  # Binary classification

# Fixed weights and biases
W1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
               [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
               [1.3, 1.4, 1.5, 1.6, 1.7, 1.8],
               [1.9, 2.0, 2.1, 2.2, 2.3, 2.4]])
b1 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]])
W2 = np.array([[0.2, 0.3, 0.4, 0.5, 0.6],
               [0.7, 0.8, 0.9, 1.0, 1.1],
               [1.2, 1.3, 1.4, 1.5, 1.6],
               [1.7, 1.8, 1.9, 2.0, 2.1],
               [2.2, 2.3, 2.4, 2.5, 2.6],
               [2.7, 2.8, 2.9, 3.0, 3.1]])
b2 = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
W3 = np.array([[0.2, 0.3, 0.4, 0.5],
               [0.6, 0.7, 0.8, 0.9],
               [1.0, 1.1, 1.2, 1.3],
               [1.4, 1.5, 1.6, 1.7],
               [1.8, 1.9, 2.0, 2.1]])
b3 = np.array([[0.1, 0.2, 0.3, 0.4]])
W4 = np.array([[0.2], [0.3], [0.4], [0.5]])
b4 = np.array([[0.1]])

# Training parameters
learning_rate = 0.1
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)

    z2 = np.dot(a1, W2) + b2
    a2 = relu(z2)

    z3 = np.dot(a2, W3) + b3
    a3 = relu(z3)

    z4 = np.dot(a3, W4) + b4
    a4 = relu(z4)

    # Compute error
    error = y - a4

    # Backpropagation
    d_a4 = error * sigmoid_derivative(a4)
    d_W4 = np.dot(a3.T, d_a4) * learning_rate
    d_b4 = np.sum(d_a4, axis=0, keepdims=True) * learning_rate

    d_a3 =  error * sigmoid_derivative(a3)
    d_W3 =  np.dot(a2.T, d_a3) * learning_rate
    d_b3 =  np.sum(d_a3, axis=0, keepdims=True) * learning_rate

    d_a2 =  error * sigmoid_derivative(a2)
    d_W2 =  np.dot(a1.T, d_a2) * learning_rate
    d_b2 =  np.sum(d_a2, axis=0, keepdims=True) * learning_rate

    d_a1 =  error * sigmoid_derivative(a1)
    d_W1 =  np.dot(X.T, d_a1) * learning_rate
    d_b1 =  np.sum(d_a1, axis=0, keepdims=True) * learning_rate

    # Update weights and biases
    W4 += d_W4
    b4 += d_b4
    W3 += d_W3
    b3 += d_b3
    W2 += d_W2
    b2 += d_b2
    W1 += d_W1
    b1 += d_b1

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.abs(error))
        print(f"Epoch {epoch}, Loss: {loss}")

# Final predictions
y_pred = a4
print("a4:", a4)
print("a3:", a3.max())
print("a2:", a2.max())
print("a1:", a1.max())
print("Final Prediction:", "Dog" if y_pred[0] == 1 else "Cat")



