import numpy as np
import neurolab as nl

# define the input file
input_file = 'letter.data'

# define the number of datapoints to be loaded from the input file
num_datapoints = 50

# string containing all the distinct characters
orig_labels = 'omandig'

# compute the number of distinct chacracters
num_orig_labels = len(orig_labels)

# define the training and testing parameters
num_train = int(0.9 * num_datapoints)
num_test = num_datapoints - num_train

# define the dataset extraction parameters
start = 6
end = -1

# creating the dataset
data = []
labels = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        # split the current line tabwise
        list_vals = line.split('\t')

        # check if the label is in our ground truth labels. if not, we should skip it
        if list_vals[1] not in orig_labels:
            continue

        # extract the current label and append it to the main list
        label = np.zeros((num_orig_labels, 1))
        label[orig_labels.index(list_vals[1])] = 1
        labels.append(label)

        # extract the character vector and append it to the main list
        cur_char = np.array([float(x) for x in list_vals[start:end]])
        data.append(cur_char)

        # exit the loop once the required dataset has been created
        if len(data) >= num_datapoints:
            break

# convert the data and labels to numpy arrays
data = np.asfarray(data)
labels = np.array(labels).reshape(num_datapoints, num_orig_labels)

# extract the number of dimensions
num_dims = len(data[0])

# create a feedforward neural network
nn = nl.net.newff([[0, 1] for _ in range(len(data[0]))],
                    [128, 16, num_orig_labels])

# set the training algorithm to gradient descent
nn.trainf = nl.train.train_gd

# train the network
error_progress = nn.train(data[:num_train,:], labels[:num_train,:], epochs=10000, show=100, goal=0.01)

# predict the output for test inputs
print('\nTesting on unknown data:')
predicted_test = nn.sim(data[num_train:, :])
for i in range(num_test):
    print('\nOriginal: ', orig_labels[np.argmax(labels[i])])
    print('predicted: ', orig_labels[np.argmax(predicted_test[i])])
