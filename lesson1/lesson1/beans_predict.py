import dataset
from matplotlib import pyplot as plt

xs, ys = dataset.get_beans(100)

plt.title("Size-Toxicity Function")
plt.xlabel("Bean size")
plt.ylabel("Toxicity")

plt.scatter(xs, ys)

# y = 0.5x
w = 0.5
y_prediction = w * xs
plt.plot(xs, y_prediction)

plt.show()