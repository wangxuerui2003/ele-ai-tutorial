import numpy as np
import matplotlib.pyplot as plt
from dataset import get_beans
from mpl_toolkits.mplot3d import Axes3D

DATASET_SIZE = 100

xs, ys = get_beans(DATASET_SIZE)

# set plot title, x-label and y-label
plt.title("Size-Toxicity Prediction")
plt.xlabel("Beans Size")
plt.ylabel("Toxicity")

# set the limit of the plot to prevent inconsistent graph size
# plt.xlim(-2, 2)
# plt.ylim(0, 2)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# draw scatter graph for beans
plt.scatter(xs, ys)

# 1 hidden layer neural network
#       /---- O -\
# X ----          ---- O ---- Y
#       \---- O -/

'''
	Naming Convention:

	wab_c
	w stands for weight
	a is the nth input parameter
	b is the nth neuron in current layer
	c is the current layer number

	bm_n
	b stands for bias
	m is the nth neuron in current layer
	n is the current layer number

	zm_n
	z is the linear function output
	m is the nth neuron in current layer
	n is the current layer number
'''

# first layer, 1st neuron
w11_1 = np.random.rand()
b1_1 = np.random.rand()

# first layer, 2nd neuron
w12_1 = np.random.rand()
b2_1 = np.random.rand()

# second layer (the output layer)
w11_2 = np.random.rand()
w21_2 = np.random.rand()
b1_2 = np.random.rand()  # 2 inputs from the hidden layer output, but only one neuron so only one bias

def forward_propagation(x):
	z1_1 = w11_1*x + b1_1
	a1_1 = sigmoid(z1_1)

	z2_1 = w12_1*x + b2_1
	a2_1 = sigmoid(z2_1)

	z1_2 = w11_2*a1_1 + w21_2*a2_1 + b1_2
	a1_2 = sigmoid(z1_2)
	return z1_1, z2_1, z1_2, a1_1, a2_1, a1_2

z1_1, z2_1, z1_2, a1_1, a2_1, a1_2 = forward_propagation(xs)

# draw the activation function
plt.plot(xs, a1_2)

# show the matplotlib plot
plt.show()

def deda(y, a):
	# e = (y-a)^2
	# deda = -2(y-a)
	return -2*(y-a)

def dadz(a):
	# a = 1/(1+e^(-z))
	# dadz = a(1-a)
	return a*(1-a)

def dzdw(x):
	# z = wx+b
	# dzdw = x
	return x

def dzdb(x):
	# z = wx+b
	# dzdb = 1
	return 1

def dzda(w):
	# z = wa + b
	# dzda = w
	return w

alpha = 0.05

# SGD
for epoch in range(1000):
	for i in range(DATASET_SIZE):
		x = xs[i]
		y = ys[i]

		# forward propagation
		z1_1, z2_1, z1_2, a1_1, a2_1, a1_2 = forward_propagation(x)

		# back propagation
		# get dedw11_2 and dedw21_2
		deda1_2 = deda(y, a1_2)  # start from the last neuron (output neuron)
		da1_2dz1_2 = dadz(a1_2)
		dz1_2dw11_2 = dzdw(a1_1)
		dz1_2dw21_2 = dzdw(a2_1)
		dedw11_2 = deda1_2 * da1_2dz1_2 * dz1_2dw11_2
		dedw21_2 = deda1_2 * da1_2dz1_2 * dz1_2dw21_2

		# get dedb1_2
		dz1_2db1_2 = dzdb(a1_1)
		dedb1_2 = deda1_2 * da1_2dz1_2 * dz1_2db1_2

		# get dz1_2dw11_1 and dz1_2dw12_1
		dz1_2da1_1 = dzda(w11_2)
		da1_1dz1_1 = dadz(a1_1)
		dz1_1dw11_1 = dzdw(x)
		dz1_2dw12_1 = dzda(w21_2)
		da2_1dz2_1 = dadz(a2_1)
		dz2_1dw12_1 = dzdw(x)
		dedw11_1 = deda1_2 * da1_2dz1_2 * dz1_2da1_1 * da1_1dz1_1 * dz1_1dw11_1
		dedw12_1 = deda1_2 * da1_2dz1_2 * dz1_2dw12_1 * da2_1dz2_1 * dz2_1dw12_1

		# get dedb1_1 and dedb2_1
		dz1_1db1_1 = dzdb(x)
		dz2_1db2_1 = dzdb(x)
		dedb1_1 = deda1_2 * da1_2dz1_2 * dz1_2da1_1 * da1_1dz1_1 * dz1_1db1_1
		dedb2_1 = deda1_2 * da1_2dz1_2 * dz1_2dw12_1 * da2_1dz2_1 * dz2_1db2_1

		# gradient descent
		w11_1 = w11_1 - alpha * dedw11_1
		w12_1 = w12_1 - alpha * dedw12_1
		b1_1 = b1_1 - alpha * dedb1_1
		b2_1 = b2_1 - alpha * dedb2_1
		w11_2 = w11_2 - alpha * dedw11_2
		w21_2 = w21_2 - alpha * dedw21_2
		b1_2 = b1_2 - alpha * dedb1_2

	if epoch % 10 != 0:
		continue
	plt.clf()
	plt.scatter(xs, ys)
	# plt.xlim(0, 1)
	# plt.ylim(0, 1.2)
	z1_1, z2_1, z1_2, a1_1, a2_1, a1_2 = forward_propagation(xs)
	plt.plot(xs, a1_2)
	plt.pause(0.01)

plt.show()