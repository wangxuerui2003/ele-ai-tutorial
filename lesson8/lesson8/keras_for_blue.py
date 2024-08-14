import dataset
import numpy as np
import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

DATASET_SIZE = 100

def single_input_single_neuron():
	X, Y = dataset.get_beans1(DATASET_SIZE)

	plot_utils.show_scatter(X, Y)

	model = Sequential()
	model.add(Dense(units=1, activation='sigmoid', input_dim=1))
	model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
	model.fit(X, Y, epochs=1000, batch_size=10)

	predictions = model.predict(X)

	plot_utils.show_scatter_curve(X, Y, predictions)

def single_input_one_hidden_layer_with_2_neurons():
	X, Y = dataset.get_beans2(DATASET_SIZE)

	plot_utils.show_scatter(X, Y)

	model = Sequential()
	model.add(Dense(units=2, activation='sigmoid', input_dim=1))
	model.add(Dense(units=1, activation='sigmoid'))
	model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
	model.fit(X, Y, epochs=5000, batch_size=10)

	predictions = model.predict(X)

	plot_utils.show_scatter_curve(X, Y, predictions)

def two_inputs_with_1_neuron():
	X, Y = dataset.get_beans(DATASET_SIZE)

	plot_utils.show_scatter(X, Y)

	model = Sequential()
	model.add(Dense(units=1, activation='sigmoid', input_dim=2))
	model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
	model.fit(X, Y, epochs=5000, batch_size=10)

	predictions = model.predict(X)

	plot_utils.show_scatter_surface(X, Y, model)

def two_inputs_with_2_neurons():
	X, Y = dataset.get_beans4(DATASET_SIZE)

	plot_utils.show_scatter(X, Y)

	model = Sequential()
	model.add(Dense(units=2, activation='sigmoid', input_dim=2))
	model.add(Dense(units=1, activation='sigmoid'))
	model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
	model.fit(X, Y, epochs=5000, batch_size=10)

	predictions = model.predict(X)

	plot_utils.show_scatter_surface(X, Y, model)

if __name__ == '__main__':
	two_inputs_with_2_neurons()