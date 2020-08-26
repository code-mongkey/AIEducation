import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# input file contraining data
input_file = 'data_singlevar_regr.txt'

# read data
data = np.loadtxt(input_file, delimiter=',')
X,y = data[:, :-1], data[:, -1]

# train and test split
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# training data
X_train, y_train = X[:num_training], y[:num_training]

# test data
X_test, y_test = X[num_training:], y[num_training:]

# create linear regressor object
regressor = linear_model.LinearRegression()

# train the model using the training sets
regressor.fit(X_train, y_train)

# predict the output
y_test_pred = regressor.predict(X_test)

# plot outputs
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())

# compute preformance metrics
print('linear regressor performance')
print('mean absolute error = ', round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print('mean squared error = ', round(sm.mean_squared_error(y_test, y_test_pred), 2))
print('median absolute error = ', round(sm.median_absolute_error(y_test, y_test_pred), 2))
print('explain variance score = ', round(sm.explained_variance_score(y_test, y_test_pred), 2))

print('R2 score = ', round(sm.r2_score(y_test, y_test_pred), 2))

# model persistence
output_model_file = 'model.pkl'

# save the model
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# load the model
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# perform prediction on test data
y_test_pred_new = regressor_model.predict(X_test)
print('\nNew mean absolute error = ', round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))