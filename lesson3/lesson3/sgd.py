import dataset
from matplotlib import pyplot as plt
from time import sleep

DATASET_SIZE = 100

xs, ys = dataset.get_beans(DATASET_SIZE)

# Initialize the plot
plt.title("Size-Toxicity Function")
plt.xlabel("Bean size")
plt.ylabel("Toxicity")

plt.scatter(xs, ys)

w = 0.5

y_prediction = w * xs

alpha = 0.05

for _ in range(100):
	for i in range(DATASET_SIZE):
		x = xs[i]
		y = ys[i]

		# a = x^2
		# b = -2xy
		# c = y^2
		# gradient = 2aw + b

		k = 2*(x ** 2)*w - 2*x*y

		# adjust w
		w = w - alpha * k

y_prediction_after_learn = w * xs
plt.plot(xs, y_prediction_after_learn, color="green")
plt.plot(xs, y_prediction, color="red")

plt.show()