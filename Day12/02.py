import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

#Load input data
text = np.loadtxt('data_simple_nn.txt')

#seperate it into datapoints and labels
data = text[:, 0:2]
labels = text[:, 2:]

#plot input data
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("input data")

#minimum and maximum values for each dimension
dim1_min, dim1_max = data[:,0].min(), data[:,0].max()
dim2_min, dim2_max = data[:,1].min(), data[:,1].max()

#define the number of neurons in the output layer
num_output = labels.shape[1]

#define a single-layer neural network
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]

nn = nl.net.newp([dim1, dim2], num_output)

#train the neural network
error_pregress = nn.train(data, labels, epochs=50, show=1, lr=0.03)

#plot the training progress
plt.figure()
plt.plot(error_pregress)
plt.xlabel("number of epochs")
plt.ylabel("training error")
plt.title("training error pregree")
plt.grid()
plt.show()

#run the classifier on test datapoints
print('\ntest rwesults:')
data_test = [[0.4, 4.3], [4.4, 0.6], [4.7, 8.1]]
for item in data_test:
    print(item, '-->', nn.sim([item])[0])

