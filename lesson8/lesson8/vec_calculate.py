import numpy as np
from dataset import get_beans, get_beans2
from plot_utils import show_scatter, show_scatter_surface
import matplotlib.pyplot as plt

DATASET_SIZE = 100

X, Y = get_beans(DATASET_SIZE)

show_scatter(X, Y)

# since there are two features, there will be 2 weights and 1 bias
# use vector for 2 weights
W = np.array([0.1, 0.1])  # w1 and w2
B = np.array([0.1])

def sigmoid(Z: np.ndarray):
	return 1/(1+(np.exp(-Z)))

def forward_propagation(X: np.ndarray):
	# X is a 2d np array
	# z = w1x1 + w2x2 + b
	# a = sigmoid(z)
	# z = w1*x1s + w2*x2s + b
	Z = X.dot(W.T)  # dot product, sum of all x * all w
	A = sigmoid(Z)
	return A

A = forward_propagation(X)

show_scatter_surface(X, Y, forward_propagation)

alpha = 0.05

for epoch in range(1000):
	for i in range(DATASET_SIZE):
		Xi = X[i]
		Yi = Y[i]

		A = forward_propagation(Xi)

		E = (Y - A) ** 2

		# dedw1 = deda * dadz * dzdw1
		# dedw2 = deda * dadz * dzdw2
		# dedb = deda * dadz * dzdb
		dEdA = -2*(Yi-A)
		dAdZ = A*(1-A)
		dZdW = Xi
		dZdB = 1
		dEdW = dEdA * dAdZ * dZdW
		dEdB = dEdA * dAdZ * dZdB

		W = W - alpha*dEdW
		B = B - alpha*dEdB

show_scatter_surface(X, Y, forward_propagation)
