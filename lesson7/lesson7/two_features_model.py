import numpy as np
from dataset import get_beans, get_beans2
from plot_utils import show_scatter, show_scatter_surface, show_surface
import matplotlib.pyplot as plt

DATASET_SIZE = 100

xs, ys = get_beans(DATASET_SIZE)

show_scatter(xs, ys)

# since there are two features, there will be 2 weights and 1 bias
w1 = 0.1
w2 = 0.1
b = 0.1

def sigmoid(z):
	return 1/(1+(np.exp(-z)))

def forward_propagation(x1s, x2s):
	# xs is a 2d np array
	# z = w1x1 + w2x2 + b
	# a = sigmoid(z)
	z = w1*x1s + w2*x2s + b
	a = sigmoid(z)
	return a

# beans size
x1s = xs[:, 0]

# beans color brightness
x2s = xs[:, 1]

a = forward_propagation(x1s, x2s)

show_scatter_surface(xs, ys, forward_propagation)

alpha = 0.05

for epoch in range(1000):
	for i in range(DATASET_SIZE):
		x = xs[i]
		y = ys[i]
		x1 = x[0]
		x2 = x[1]

		a = forward_propagation(x1, x2)

		e = (y - a) ** 2

		# dedw1 = deda * dadz * dzdw1
		# dedw2 = deda * dadz * dzdw2
		# dedb = deda * dadz * dzdb
		deda = -2*(y-a)
		dadz = a*(1-a)
		dzdw1 = x1
		dzdw2 = x2
		dzdb = 1
		dedw1 = deda * dadz * dzdw1
		dedw2 = deda * dadz * dzdw2
		dedb = deda * dadz * dzdb

		w1 = w1 - alpha*dedw1
		w2 = w2 - alpha*dedw2
		b = b - alpha*dedb

show_scatter_surface(xs, ys, forward_propagation)
