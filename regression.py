# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('name.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Taking care of missing data
'''from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])'''

# Encoding categorical data
# Encoding the Independent Variable
# here 0 index is the column where categorical data is present
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)'''
#check dummy trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""


# Applying Kernel PCA
# Scaling req
'''from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf')
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)'''




# Applying LDA
# Scaling req
'''from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)'''








#REGRESSION

#Simple Linear Regression
# Fitting Simple Linear Regression to the Training set
'''from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)'''

# Predicting the Test set results
'''y_pred = regressor.predict(X_test)'''

#predicting new result having x = 78
'''regressor.predict(78)'''

# Visualising the Training set results
'''plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Title (Training set)')
plt.xlabel('X label')
plt.ylabel('y Label')
plt.show()'''

# Visualising the Test set results
'''plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Title (Test set)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()'''


# Multiple linear regression
# Fitting Multiple Linear Regression to the Training set
'''from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)'''

# Predicting the Test set results
'''y_pred = regressor.predict(X_test)'''
# cannot visualise



# BACKWARD ELIMINATION
# We are infact creating a brand new model here (OLS implies linear regression) hence no need
# to implement splitting as we just need to figure out the useless variables which are better found
# when all the data is present, hence we dont split and use the entire X and y
'''import statsmodels.formula.api as sm
X = np.append(arr= np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5]] #entire X
reg_OLS = sm.OLS(endog = y, exog = X_opt).fit()
print(reg_OLS.summary())'''
# find the var with Significance Level above certain threshold (e.g 0.05)
# remove it from X_opt and rerun the summary, find again and remove till all are below SL
# see that adjusted RSquared always increase, model is ready




# Polynomial Regression
'''from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
regressor = LinearRegression()
regressor.fit(X_poly, y)'''

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
'''X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Title')
plt.xlabel('X')
plt.ylabel('y')
plt.show()'''

# Predicting a new result with Polynomial Regression
# polyreg fittransform converts this number to an array with powers of the number
'''regressor.predict(poly_reg.fit_transform(6.5))'''



# SVR regression
# FEATURE SCALING NEEDED
# Fitting SVR to the dataset
'''from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)'''

# Predicting a new result
'''y_pred = regressor.predict(6.5)'''
# de-scale the predicted value so we can comprehend it
'''y_pred = sc_y.inverse_transform(y_pred)'''

# Visualising the SVR results (for higher resolution and smoother curve)
'''X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Title')
plt.xlabel('X')
plt.ylabel('y')
plt.show()'''


# Decision tree
# Fitting Decision Tree Regression to the dataset
'''from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)'''

# Predicting a new result
'''y_pred = regressor.predict(6.5)'''

# Visualising the Decision Tree Regression results (higher resolution)
# SAME AS ABOVE


# Random Forest
# Fitting Random Forest Regression to the dataset
'''from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X, y)'''

# Predicting a new result
'''y_pred = regressor.predict(6.5)'''

# VISUALISING IS SAME




# Applying k-Fold Cross Validation
'''from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()'''



# Applying Grid Search to find the best model and the best parameters
'''from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_'''


# REGRESSION TEMPLATE OVER


