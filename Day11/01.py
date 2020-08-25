import numpy as np

dim1 = np.array([1,2,3,4])
print(dim1)
print(dim1.ndim)
print(dim1.shape)
print(dim1.size)

dim2 = np.array([[1,2,3,4],[1,2,3,4],[1,2,3,4]])
print(dim2)
print(dim2.ndim)
print(dim2.shape)
print(dim2.size)

def init_network():
    network = {}
    network['w1']=np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1']=np.array([0.1,0.2,0.3])
    network['w2']=np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2']=np.array([0.1,0.2])
    network['w3']=np.array([[0.1,0.3],[0.2,0.4]])
    network['b3']=np.array([0.1,0.2])
    return network

def forward(network,x):
    w1,w2,w3 = network['w1'],network['w2'],network['w3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    
    neuron1 = np.dot(x, w1) + b1
    activation1 = sigmoid(neuron1)
    print(activation1)

    neuron2 = np.dot(activation1, w2) + b2
    activation2 = sigmoid(neuron2)
    print(activation2)

    neuron3 = np.dot(activation2, w3) + b3
    print_result = neuron3

    return print_result

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

network=init_network()
x=np.array([1.0, 0.5])
y=forward(network, x)
print(y)

    