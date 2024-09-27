import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries 2.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

# The standard scale class that will perform feature scaling
# expect one unique format in its input, which is a 2D array
#
# If the input is a 1D array, we have to transform It
y = y.reshape(len(y), 1)

# Feature Scaling
#
# In SVR we have to apply feature scaling, as we don't have coefficinets
# multiplying the independent variables that mitigates the outlier values
# 
# Here we don't have a split of the dataset between Training set and Test set.
# So we apply excpetionally feature scaling on the whole set X
# Dependent variables have a wide set of values (are not just 0 or 1)
# so in this case we apply feature scaling to y as well.
# This is the most common situation.
#
# We don't apply feature scaling:
# - to dummy variable resulting from one-hot encoding
# - to dependent variable takes binary values
# We apply feature scaling:
# - to dependent variables when they take very differet value ranges compared to other features
# - if we split dataset between Training set and test set, we have to apply
# - features scaling after the split
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# Training the SVR model on the whole dataset
#
# We use the class SVR from svm module provided by sklearn library
# In SVR we have to specify the kernel, which can either learn some 
# linear relationship (linear kernel) or non-linear relationships
# (non-linear kernels) such as RBF radial basis (Gaussian RBF Kernel)
# or polynomial kernel
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

# Predicting a new result with SVR
#
# We need to reverse the feature scaling both for X and y
# We use the inverse_transform method
sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1,1))

# Visualising the SVR results
#
# # Visualising the SVR results
#
# We apply inverse transformation to get the values back to the original scale
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1, 1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()