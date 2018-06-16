# Convolutional Neural Network

import os

# To use CPU, comment out to use GPU (FOR LARGE NEURAL NETS)
# RECOMMENDED TO USE GPU, SHIT GETS TIGHT
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Part 1 - Building the CNN

# Saving the model

'''classifier.save('anyname.h5')'''

# Loading the model
'''from keras.models import load_model

classifier = load_model('anyname.h5')'''

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# first three inputs are the number of feature maps, dimension of feature maps
# input shape is the expected image size in pixels with third input being channels i.e. 1 for b/w and 3 for colored
# activation must be relu or leaky relu
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
# pooling must be done after every convulution
# pool size is the size of pooling selector, 2,2 IS OPTIMAL
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
# no changes just remove input shape as it knows it from previous pooling layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a third convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a fourth convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# make it into input elements
classifier.add(Flatten())

# Step 4 - Full connection
# SEE ANN FOR TIPS
# no need to add input dims as flatten does that, directly ass the hidden layers
# take units > 100
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 128, activation = 'relu'))
# output node
# go softmax + units to no. of categories of multi category
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
# go categorical_crossentropy for multi
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# this is to augment the images, flipping, scaling them 
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

# import images, target size must match up with input shape up top
# switch classmode if multi category
# INCREASE TARGET SIZE FOR GREAT BUMP IN ACCURACY BUT SHIT WILL RUN LONG
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# samples per epoch is the total train test size
# nb val samples is the test set number to test model after each epoch
classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)




'''# Part 3 - Tuning & Evaluating'''
# see any dropout functionality and multiple category functionality from above, not included here




# NO KFOLD as it validates after every epoch see last 10 - 15 epochs to get an idea





# GRID SEARCH

# THIS SHIT WILL TAKE FCKN DAYS TO RUN

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