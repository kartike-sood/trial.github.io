# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 11:27:20 2021

@author: Kartike Sood
"""

import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

data = pd.read_csv("Churn_Modelling.csv")

X = data.iloc[ : , 3 : 13]
Y = data.iloc[ : , 13]

# data wrangling
geography = pd.get_dummies(X["Geography"], drop_first = True)
gender = pd.get_dummies(X["Gender"], drop_first = True)

X = pd.concat([X, geography, gender], axis = 1)
X.drop(["Geography", "Gender"], axis = 1, inplace = True)


# Function to create ANN model
# def create_my_model():
#     my_model = Sequential()
#     my_model.add(Dense(units = 6, input_dim = 11, activation = "relu"))
#     my_model.add(Dense(units = 1, activation = "sigmoid"))
    
#     # Compiling the model
#     my_model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
#     return my_model

# model = KerasClassifier(build_fn = create_my_model)

# # Defining the grid search parameters
# batchSize = [40, 60, 80, 100]
# epochs = [15, 20, 30]

# parameter_grid = dict(batch_size = batchSize, epochs = epochs)

# my_grid = GridSearchCV(estimator = model, param_grid = parameter_grid, n_jobs = -1, cv = 2)
# grid_result = my_grid.fit(X, Y)

# Function to create ANN model
def create_my_model(layers, activation):
    model = Sequential()
    
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim = X.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))
    model.compile(optimizer = "adam", metrics = ["accuracy"], loss = "binary_crossentropy")
    
    return model  

model = KerasClassifier(build_fn = create_my_model, verbose = 0)

# Defining the grid search parameters
layers = [(20,), (40, 20), (45, 30, 15)]
activations = ["sigmoid", "relu"]


param_grid = dict(layers = layers, activation = activations, batch_size = [10], epochs = [30])

my_grid = GridSearchCV(estimator = model, param_grid = param_grid, cv =5)
grid_result = my_grid.fit(X, Y)



# summary results
print("Best : %f using %s" % (grid_result.best_score_, grid_result.best_params_))

