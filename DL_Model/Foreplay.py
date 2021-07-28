import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv("Churn_Modelling.csv")

# There are 13 features in total
X = dataset.iloc[ : , 3 : 13]

# label
Y = dataset.iloc[ : , 13]

# In the dataset, geography and gender fields can be
# used as Groups as they are categorical features, so
# we should convert them to dummy variables

geography = pd.get_dummies(X["Geography"], drop_first = True)
gender = pd.get_dummies(X["Gender"], drop_first = True)

# Since we have created our dummy fields its time to 
# concatenate them with the original dataset and drop the
# categorical columns

X = pd.concat([X, geography, gender], axis = 1)
X.drop(['Geography', 'Gender'], axis = 1, inplace = True)

# Splitting the dataset into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size = 0.2,
                                                    random_state = 42)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


"""Let's make the ANN !!"""
# import tensorflow as tf
import keras
from keras.models import Sequential # This is used to make any type of Neural Network, be it ANN, CNN or RNN, anything
from keras.layers import Dense # This is used to create input, hidden and output layers
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout

# Initialising the NN using Sequential()
classifier = Sequential()

"""
1. units -> output_dim
2. kernel_initializer -> initialiser of weights
3. activation -> activation function to be used in the respective layer
"""
# Adding the input layer and the first Hidden layer
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", 
                                  activation = "relu", input_dim = 11)) 
# input_dim = 11, because we have 11 independent variables in our data

classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", 
                                  activation = "relu")) 

# Adding the second hidden layer of neurons
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform",
                                                  activation = "relu"))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform", # glorot_uniform is used because it works better in case of ouput layer
                                                  activation = "sigmoid"))

""" Compiling the ANN """
# If our output has only 0 and 1, then Keras documentation tells us
# to use Binary Cross Entropy, otherwise Categorical Cross Entropy
 
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", 
                                                    metrics = ["accuracy"])


# Finally fitting our data into the classifier model
model_history = classifier.fit(X_train, Y_train, validation_split = 0.33,
                                                batch_size = 10, epochs = 100)

Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

from sklearn.metrics import accuracy_score, confusion_matrix
score = accuracy_score(Y_pred, Y_test)
cm = confusion_matrix(Y_test, Y_pred)