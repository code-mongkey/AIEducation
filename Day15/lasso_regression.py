import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

redwine = pd.read_csv("data/winequality-red.csv", sep=";" )
redwine["type"]="red"
whitewine = pd.read_csv("data/winequality-white.csv", sep=";" )
whitewine["type"]="white"

#print(redwine.head(10))
#print(whitewine.head(10))

wine = redwine.append(whitewine)
wine.columns=wine.columns.str.replace(" ", "_")

print(wine.head())

from sklearn.linear_model import LinearRegression
model=LinearRegression(fit_intercept=True)
X=wine.drop(["type", "quality"], axis=1)

print(X.head())
print(X.shape)

y=wine.quality

print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(model.fit(X_train, y_train))
print(model.coef_)
print(X_train.columns)
print(model.intercept_)

newdata=np.array([6.3, 0.3, 0.34, 1.6, 0.049, 14, 132, 0.994, 3.3, 0.49, 9.5])
print(model.predict(np.reshape(newdata, (1, 11))))

y_pred=model.predict(X_test)
print(y_pred.shape)

#def rmse(y_real, y_pred):
# return np.sqrt(np.mean(y_real, y_pred)**2)
#np.round(rmse(y_real, y_pred), 2)

from sklearn.metrics import mean_squared_error
print(np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
from sklearn.linear_model import Ridge

model = Ridge(alpha=0.05)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print(np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
from sklearn.linear_model import Lasso
model = Lasso(alpha=0.05)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)
print(np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3))
from sklearn.linear_model import Ridge
#model_name="ridge"
#
ridge = Ridge(alpha=0.05)
ridge.fit(X_train, y_train)
print(ridge.fit(X_train, y_train))
y_pred=ridge.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
print(rmse)
fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(111)
alpha=0.5
ridge = Ridge(alpha=alpha)
ridge.fit(X_train, y_train)
#print(ax)
coef = pd.Series(data=ridge.coef_, index=X_train.columns).sort_values()
ax.bar(coef.index, coef.values)
ax.set_xticklabels(coef.index, rotation=90)
ax.set_title("alpha={}".format(alpha))
#print(ax.bar, ax.set_xticklabels, ax.set_title)
plt.show()
from sklearn.linear_model import Lasso
model_name="lasso"
alpha=2
fig = plt.figure(figsize=(6,3))
ax = fig.add_subplot(111)
lasso = Lasso(alpha=alpha)
lasso.fit(X_train, y_train)
y_pred=lasso.predict(X_test)
from sklearn.metrics import mean_squared_error
rmse = np.round(np.sqrt(mean_squared_error(y_test, y_pred)), 3)
coef = pd.Series(data=lasso.coef_, index=X_train.columns).sort_values()
ax.bar(coef.index, coef.values)
ax.set_xticklabels(coef.index, rotation=90)
ax.set_title("{0}: alpha={1}, rmse={2}".format(model_name, alpha, rmse))
plt.show()
