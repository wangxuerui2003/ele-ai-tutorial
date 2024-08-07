import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataset import get_beans

# Function to calculate the loss (Mean Squared Error)
def compute_loss(xs, ys, w, b):
    y_pred = w * xs + b
    return np.mean((ys - y_pred) ** 2)

# Generate dataset
DATASET_SIZE = 100
xs, ys = get_beans(DATASET_SIZE)

# Initial parameters
w = 0.6
b = 0.1
a = 0.05

# Set up 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title("SGD Progress on Size-Toxicity Prediction")
ax.set_xlabel('Weight (w)')
ax.set_ylabel('Bias (b)')
ax.set_zlabel('Loss')
ax.set_zlim(0, 0.5)

# Generate a meshgrid for the loss surface
w_range = np.linspace(0, 1, 50)
b_range = np.linspace(0, 1, 50)
W, B = np.meshgrid(w_range, b_range)
Z = np.array([compute_loss(xs, ys, w, b) for w, b in zip(np.ravel(W), np.ravel(B))])
Z = Z.reshape(W.shape)

# Plot the loss surface
ax.plot_surface(W, B, Z, alpha=0.6, cmap='viridis')

# Initial point on the plot
point = ax.scatter(w, b, compute_loss(xs, ys, w, b), color='r', s=100)

# SGD
for epoch in range(100):
    for i in range(DATASET_SIZE):
        x = xs[i]
        y = ys[i]

        # Calculate gradients
        dw = 2 * (x ** 2) * w + 2 * x * b - 2 * x * y
        db = 2 * b + 2 * x * w - 2 * y

        # Update weights and bias
        w -= a * dw
        b -= a * db

        # Update the point on the plot
        point.remove()
        point = ax.scatter(w, b, compute_loss(xs, ys, w, b), color='r', s=100)
        plt.pause(0.01)

plt.show()
