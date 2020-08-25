import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

#Load input data
text = np.loadtxt('data_perceptron.txt')

#Separate datapoints and labels
data = text[:, :2]
labels = text[:, 2].reshape((text.shape[0], 1))

#plot input data
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("input data")

#define minimum and maximum values for each dimension
dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1

#number of neurons in the output layer
num_output = labels.shape[1]

#define a perceptron with 2input neurons (because we have 2 dimensions in the input data)
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
perceptron = nl.net.newp([dim1, dim2], num_output)

#train the perceptron using the data
error_progress = perceptron.train(data, labels, epochs=1000, show=20, lr=0.03)

#plot the training progress
plt.figure()
plt.plot(error_progress)
plt.xlabel("number of epochs")
plt.ylabel("training error")
plt.title("training error progress")
plt.grid()
plt.show()
