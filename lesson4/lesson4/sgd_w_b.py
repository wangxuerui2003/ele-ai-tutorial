import numpy as np
import matplotlib.pyplot as plt
from dataset import get_beans
from mpl_toolkits.mplot3d import Axes3D

DATASET_SIZE = 100

xs, ys = get_beans(DATASET_SIZE)

# initial weight
w = 0.1

# initial learning rate alpha
a = 0.05

# initial bias
b = 0.05

# set plot title, x-label and y-label
plt.title("Size-Toxicity Prediction")
plt.xlabel("Beans Size")
plt.ylabel("Toxicity")

# set the limit of the plot to prevent inconsistent graph size
plt.xlim(0, 1.1)
plt.ylim(0, 1.6)

# draw scatter graph for beans
plt.scatter(xs, ys)

# initial prediction function graph
y_pre = w*xs + b

# draw the initial prediction line
plt.plot(xs, y_pre)

# show the matplotlib plot
plt.show()


# SGD
for epoch in range(100):
	for i in range(DATASET_SIZE):
		x = xs[i]
		y = ys[i]

		# a = x^2
		# b = -2xy
		# c = y^2

		# dw = 2(x^2)w + 2xb - 2xy
		dw = 2*(x ** 2)*w + 2*x*b - 2*x*y
		# db = 2b + 2xw - 2y
		db = 2*b + 2*x*w - 2*y

		# adjust w and b
		w = w - a*dw
		b = b - a*db

	plt.clf()
	plt.scatter(xs, ys)
	plt.xlim(0, 1)
	plt.ylim(0, 1.5)
	y_pre = w*xs + b
	plt.plot(xs, y_pre)
	plt.pause(0.05)

plt.show()