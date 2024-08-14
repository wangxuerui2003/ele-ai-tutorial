import dataset
import numpy as np
import plot_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

DATASET_SIZE = 100

X, Y = dataset.get_beans(DATASET_SIZE)

plot_utils.show_scatter(X, Y)

# Deep Neural Network with 3 hidden layers

model = Sequential()
model.add(Dense(units=8, activation='relu', input_dim=2))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=0.05), metrics=['accuracy'])
model.fit(X, Y, epochs=5000, batch_size=10)

# predictions = model.predict(X)

plot_utils.show_scatter_surface(X, Y, model)