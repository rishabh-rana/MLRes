# Recurrent Neural Network

# To use CPU, comment out to use GPU (FOR LARGE NEURAL NETS) IF YOU HAVE TENSORFLOW_GPU INSTALLED
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# these columns are the indicators, first one making the timesteps
# other indicators just help predict the outcome (without forming timesteps)
# we can take multiple columns as indicators the code is optimized to do all shit automatically iloc the extra column
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
# this is normalisation, not stndardisation as its recommended in RNN, scale other features also if you add them
from sklearn.preprocessing import MinMaxScaler
# this automatically scales all indicators
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    # i-60:i gets 2 dimensions, ,: gets another, this another dimension holds all the indicators
    X_train.append(training_set_scaled[i-60:i, :])
    # we use 0:1 to get the right dimension (make y an array)
    y_train.append(training_set_scaled[i, 0:1])
X_train, y_train = np.array(X_train), np.array(y_train)





# Part 2 - Building the RN

# Saving the model

'''regressor.save('anyname.h5')'''

# Loading the model
'''from keras.models import load_model

regressor = load_model('anyname.h5')'''

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
# take more units for complex shit
# return sequences default is false, set it to true if another LSTM layer is to be added
# add the last 2 shapes in input shape, 1st shape is taken automatically
# input shape is added at first layer only as the return sequences passes this data to other layers automatically
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
'''regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))'''

# Adding a third LSTM layer and some Dropout regularisation
'''regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))'''

# Adding a fourth LSTM layer and some Dropout regularisation
# return sequences defaulted to false
# last layer
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
# loss is mean squared for regression
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
'''dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values'''

# Getting the predicted stock price of 2017
'''dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, :])
X_test = np.array(X_test)
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)'''

# Visualising the results
'''plt.plot(real_stock_price, color = 'red', label = 'Real')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted')
plt.title('Title')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()'''











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
    classifier.compile(optimizer = 'adam', loss = 'mean_squared_error')
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
    classifier.compile(optimizer = optimizer, loss = 'mean_squared_error')
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           # scoring here is different NOT ACCURACY AS IN CLASSIFICATION
                           scoring = 'neg_mean_squared_error',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_'''








