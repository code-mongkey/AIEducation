import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

#generate some training data
min_val = -15
max_val = 15
num_points = 130

x = np.linspace(min_val, max_val, num_points)
y = 3 * np.square(x) + 5
y /= np.linalg.norm(y)

#create data and labels
data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

#plot input data
plt.figure()
plt.scatter(data, labels)
plt.xlabel("dimension 1")
plt.ylabel("dimension 2")
plt.title('input data')

# define a multilayer neural network with 2 hidden layers
# first hidden layer consists of 10 neurons
# second hidden layer consists of 6 neurons
# output layer consists of 1 neuron

nn = nl.net.newff([[min_val, max_val]], [10, 6, 1])

#set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd

#train the neural network
error_progress = nn.train(data, labels, epochs=5000,  show=100, goal=0.001)

#run the neural network on training datapoints
output = nn.sim(data)
y_pred = output.reshape(num_points)

#plot training error
plt.figure()
plt.plot(error_progress)
plt.xlabel('number of epochs')
plt.ylabel('error')
plt.title('training error progress')

#plot the output
x_dense = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size, 1)).reshape(x_dense.size)
plt.figure()
plt.plot(x_dense, y_dense_pred, '-', x, y, '.', x, y_pred, 'p')
plt.title('actual vs predicted')
plt.show()
