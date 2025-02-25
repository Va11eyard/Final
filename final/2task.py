import numpy as np
import pandas as pd
from scipy.special import expit

# Sigmoid function
def sigmoid(z):
    return expit(z)

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (-1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

# Gradient descent function
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    cost_history = []
    for _ in range(iterations):
        gradient = (1/m) * (X.T @ (sigmoid(X @ theta) - y))
        theta -= alpha * gradient
        cost_history.append(compute_cost(X, y, theta))
    return theta, cost_history

# Load dataset from CSV
data = pd.read_csv('Question3_Final_CP.csv')
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values  # Target

m, n = X.shape
X = np.c_[np.ones(m), X]  # Add bias term
y = y.reshape(-1, 1)

# Initialize theta
theta = np.zeros((n + 1, 1))
alpha = 0.01

# Run gradient descent for different iterations
iterations_list = [10, 100, 1000]
results = {}

for iters in iterations_list:
    theta_opt, cost_history = gradient_descent(X, y, theta.copy(), alpha, iters)
    results[iters] = (round(cost_history[-1], 3), round(np.max(theta_opt), 2))

# Display results
print("# Iterations | Cost Function | Max Theta Value")
for k, v in results.items():
    print(f"n={k} | {v[0]} | {v[1]}")

# Run for 10,000 iterations and predict
theta_opt, _ = gradient_descent(X, y, theta.copy(), alpha, 10000)
y_pred = sigmoid(X @ theta_opt) >= 0.7
num_ones = np.sum(y_pred[:20])
print(f"Number of ones in first 20 predictions: {num_ones}")