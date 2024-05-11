# first neural network with keras tutorial

from numpy import loadtxt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# loading data:
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input X and output y variables
X = dataset[:,0:8]
y = dataset[:,8]

# create Sequential model
model = Sequential()
# the convention is slightly confusing, the shape of input is one of the arguments. so the first model.add is doing two things: 1. declaring input / visible layer, and the first hidden layer
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

# choose a loss function, specify the stochastic gradient descent algorithm "adam" and specify what metric to report
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# start training using the fit() function on the model
# use a relatively small epoch and batch size
model.fit(X, y, epochs=150, batch_size=10)

# the evaluate function returns a list with two variables, 1: loss; 2: accuracy of model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %2f' % (accuracy*100))
