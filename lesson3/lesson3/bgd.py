import dataset
from matplotlib import pyplot as plt
from time import sleep
import numpy as np

DATASET_SIZE = 100

xs, ys = dataset.get_beans(DATASET_SIZE)

# Initialize the plot
plt.title("Size-Toxicity Function")
plt.xlabel("Bean size")
plt.ylabel("Toxicity")

plt.scatter(xs, ys)

w = 0.5

y_prediction = w * xs

alpha = 0.1

for _ in range(100):
	for i in range(DATASET_SIZE):
		# a = x^2
		# b = -2xy
		# c = y^2
		# gradient = 2aw + b
		k = (2*(np.sum(xs ** 2))*w - 2*np.sum(xs*ys)) / DATASET_SIZE

		# adjust w
		w = w - alpha * k

		plt.clf()
		plt.scatter(xs, ys)
		y_pre = w * xs
		plt.plot(xs, y_pre)
		plt.pause(0.01)

y_prediction_after_learn = w * xs
plt.plot(xs, y_prediction_after_learn, color="green")
plt.plot(xs, y_prediction, color="red")

plt.show()