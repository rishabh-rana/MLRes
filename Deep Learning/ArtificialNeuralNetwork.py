# Artificial Neural Network

# Tip to run KFold or GridSearch, use multiple laptops or google colab at the same time with different parameters/ random state it will give same results quickly

'''# Part 1 - Data Preprocessing same shit'''

# Importing the libraries
import numpy as np
import pandas as pd

# To use CPU, comment out to use GPU (FOR LARGE NEURAL NETS) IF YOU HAVE TENSORFLOW_GPU INSTALLED
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Importing the dataset
dataset = pd.read_csv('name.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Encoding categorical data
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# dummy trap
X = X[:, 1:]'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling  REQUIRED ALWAYS FOR NN
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)















'''# Part 2 - CREATE ANN'''

# Saving the model

'''classifier.save('anyname.h5')'''

# Loading the model
'''from keras.models import load_model

classifier = load_model('anyname.h5')'''

import keras

# initialise the model

from keras.models import Sequential

# add neurons to model

from keras.layers import Dense

# prevent overfitting by deactivating neurons randomly over epochs

from keras.layers import Dropout

# Initialising the ANN

classifier = Sequential()

# Adding the input layer and the first hidden layer
# input dim is the input layer = number of IV
# units give numbe rof neurons in hidden layer
# relu gives rectifier function

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# classifier.add(Dropout(p = 0.1))
# above line adds overfitting support, p gives % of neurons to block, normal range is 0.1 to 0.4 ; 0.5 max

# Adding the second hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# classifier.add(Dropout(p = 0.1))

# Adding the output layer


# IMP #
'''imp'''
# if you want a classifier that predicts multiple categories i.e. not binary outcome :
# change units to the number of categories AND change actvation to softmax

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN

# loss = categorical_crossentropy for multiple categories

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
# batch size : 25, 32 powers of 2 are recommended numbers, we can stretch epochs to about 500

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

















'''# Part 3 - Making predictions and evaluating the model'''

# Predicting the Test set results

y_pred = classifier.predict(X_test)

# sigmoid gives probability, convert it to true/false

y_pred = (y_pred > 0.5)

# Predicting a single new observation given below
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
# np.array allows us to enter data in form of array, we cant enter it directly
'''new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)'''

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
















'''# Part 4 - Tuning & Evaluating'''
# see any dropout functionality and multiple category functionality from above, not included here

# K FOLD

'''
# to merge sklearn and keras
from keras.wrappers.scikit_learn import KerasClassifier
# same shit, just simple k fold
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
# function to embed into KerasClassifier to build classifier each time
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
# this classifier is just like a normal one, use it as any classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
# cv is number of folds, n_jobs dont work on windows
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()'''








# GRID SEARCH

# Dropout Regularization to reduce overfitting if needed


'''
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
# to change any hyperparameters, we need to pass it as an argument like optimizer
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    # the argument optimizer acts asvariable and diffeent models are made in grid search
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_'''