import numpy as np
import matplotlib.pylab as plt

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b=-0.7
    tmp=np.sum(w*x)+b
    print(tmp)
    if tmp <= 0:
        return 0
    else:
        return 1
    
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x)+b
    print(tmp)
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x)+b
    print(tmp)
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y = AND(s1, s2)
    return y

x1, x2 = 1.0, 0.0
y1, y2 = AND(x1,x2), NAND(x1, x2)
y3, y4 = OR(x1, x2), XOR(x1, x2)

print(y1, y2, y3, y4)

X=np.array([1,2,3])
W=np.array([[1,3],[2,4],[5,8]])
y=np.dot(X,W)
print(y)

#def step_function(x):
#    return np.array(x > 0, dtype=np.int)
#def relu(x):
#    return np.maximum(0, x)
#def sigmoid(x):
#    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    z = exp_a / sum_exp_a
    return z

#a = np.array([0.3, 2.9, 4.0])
#y = softmax(a)
#print(y)
#print(np.sum(y))

x = np.arange(-5.0, 5.0, 0.1)
y = softmax(x)
print(y)
print(np.sum(y))
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

