import numpy as np
import matplotlib.pyplot as plt
from dataset import get_beans
from mpl_toolkits.mplot3d import Axes3D

DATASET_SIZE = 100

xs, ys = get_beans(DATASET_SIZE)

# initial weight
w = 0.6

# initial learning rate alpha
a = 0.05

# initial bias
b = 0

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

# construct a Axes3D object from plt.figure
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')
ax3d.set_zlim(0, 2)

# generate set of w and b values for drawing error functions
ws = np.arange(-1, 2, 0.1)
bs = np.arange(-2, 2, 0.1)

# get error function points for all bs and ws

for b in bs:
	es = []
	for w in ws:
		y_pre = w*xs + b
		e = np.sum((y_pre - ys) ** 2) / DATASET_SIZE
		es.append(e)
	ax3d.plot(ws, es, b, zdir='y')
	# or
	# ax3d.plot(ws, b, es)

# show the matplotlib plot
plt.show()
