import dataset
from matplotlib import pyplot as plt
from time import sleep

DATASET_SIZE = 100

xs, ys = dataset.get_beans(DATASET_SIZE)

# Initialize the plot
fig, ax = plt.subplots(figsize=(8, 6))
plt.ion()  # Enable interactive mode
line, = ax.plot([], [], 'r-')  # Initialize an empty line

ax.set_ylim(0, 1.5)
ax.set_xlim(0, 1.5)

def draw_line(x_data, y_data):
    line.set_xdata(x_data)
    line.set_ydata(y_data)
    ax.relim()  # Recalculate limits
    ax.autoscale_view()  # Rescale the view
    plt.draw()  # Redraw the plot
    plt.pause(0.1)

plt.title("Size-Toxicity Function")
plt.xlabel("Bean size")
plt.ylabel("Toxicity")

plt.scatter(xs, ys)

alpha = 0.05
w = 0.5
for i in range(50):
    sleep(0.2)
    for j in range(DATASET_SIZE):
        x = xs[j]
        y = ys[j]
        y_pre = w * x
        error = y - y_pre
        w = w + alpha * error * x
    y_prediction = w * xs
    draw_line(xs, y_prediction)

y_prediction = w * xs

plt.plot(xs, y_prediction)

plt.ioff()
plt.show()
