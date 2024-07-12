import numpy as np

def get_beans(counts):
	random_slope = np.random.rand() + 1
	print(random_slope)
	xs = np.random.rand(counts)
	xs = np.sort(xs)
	ys = [random_slope*x+np.random.rand()/10 for x in xs]
	return xs,ys

