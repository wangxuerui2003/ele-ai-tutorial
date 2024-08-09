import numpy as np
import matplotlib.pyplot as plt
from dataset import get_beans
from mpl_toolkits.mplot3d import Axes3D

DATASET_SIZE = 100

xs, ys = get_beans(DATASET_SIZE)

# initial weight
w = 0.1

# initial learning rate alpha
alpha = 1

# initial bias
b = 0.1

# set plot title, x-label and y-label
plt.title("Size-Toxicity Prediction")
plt.xlabel("Beans Size")
plt.ylabel("Toxicity")

# set the limit of the plot to prevent inconsistent graph size
# plt.xlim(-2, 2)
# plt.ylim(0, 2)

# draw scatter graph for beans
plt.scatter(xs, ys)

# prediction
z = w*xs + b

# activation function
a = 1 / (1 + np.exp(-z))

# draw the activation function
plt.plot(xs, a)

# show the matplotlib plot
plt.show()


# SGD
for epoch in range(100):
	for i in range(DATASET_SIZE):
		x = xs[i]
		y = ys[i]

		# MSE = (y - sigmoid(wx + b))^2
		z = w*x + b
		a = 1 / (1 + np.exp(-z))
		e = (y - a) ** 2

		deda = -2*(y - a)
		dadz = a*(1 - a)
		dzdw = x

		dedw = deda * dadz * dzdw

		dzdb = 1

		dedb = deda * dadz * dzdb

		w = w - alpha * dedw
		b = b - alpha * dedb

	if epoch % 10 != 0:
		continue
	plt.clf()
	plt.scatter(xs, ys)
	plt.xlim(0, 1)
	plt.ylim(0, 1.2)
	z = w*xs + b
	a = 1/(1 + np.exp(-z))
	plt.plot(xs, a)
	plt.pause(0.01)

plt.show()