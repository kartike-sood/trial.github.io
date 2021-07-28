# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Tue Jul 13 21:48:24 2021)---
test_list_tour = [1, 2, 3, 4, 5]
test_dict_tour = {'a': 1, 'b': 2}
import pandas as pd
runfile('C:/Users/Lenovo/.spyder-py3/temp.py', wdir='C:/Users/Lenovo/.spyder-py3')
print("Python")
print("Python")
runfile('C:/Users/Lenovo/.spyder-py3/temp.py', wdir='C:/Users/Lenovo/.spyder-py3')
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("Churn_Modelling.csv")
runcell(0, 'C:/Users/Lenovo/.spyder-py3/temp.py')
geography = pd.get_dummies(X["Geography"])
geography = pd.get_dummies(X["Geography"], drop_first = true)
geography = pd.get_dummies(X["Geography"], drop_first = True)
gender = pd.concat(X["Gender"], drop_first = True)
gender = pd.get_dummies(X["Gender"], drop_first = True)
X = pd.concat([X, geography, gender], axis = 1)

X = pd.concat([X, geography, gender], axis = 1)
X = X.drop(["Geography", "Gender"], axis = 1)
X = X.drop(['Geography', 'Gender'], axis = 1, drop = True)
X = X.drop(['Geography', 'Gender'], axis = 1, changes = True)
X = X.drop(['Geography', 'Gender'], axis = 1, inplace = True)
runcell(0, 'C:/Users/Lenovo/.spyder-py3/temp.py')
X = X.drop(['Geography', 'Gender'], axis = 1)
runcell(0, 'C:/Users/Lenovo/.spyder-py3/temp.py')
X.drop(['Geography', 'Gender'], axis = 1, inplace = True)
runcell(0, 'C:/Users/Lenovo/.spyder-py3/temp.py')
import keras
import tensorflow as tf

## ---(Wed Jul 14 00:30:38 2021)---
runfile('C:/Users/Lenovo/.spyder-py3/Foreplay.py', wdir='C:/Users/Lenovo/.spyder-py3')
import tensorflow as tfl
python
import tensorflow as tf
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Foreplay.py')

## ---(Wed Jul 14 02:25:09 2021)---
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Foreplay.py')
import tensorflow as tf
# import keras
import tensorflow as tf
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Foreplay.py')

## ---(Wed Jul 14 09:31:57 2021)---
runcell(0, 'C:/Users/Lenovo/.spyder-py3/untitled0.py')

## ---(Wed Jul 14 13:42:56 2021)---
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Foreplay.py')
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Foreplay.py')

## ---(Wed Jul 14 21:53:58 2021)---
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Foreplay.py')

## ---(Wed Jul 14 22:06:22 2021)---
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Foreplay.py')

classifier = Sequential()

# Adding the input layer and the first Hidden layer
classifier.add(Dense(unite = 6, kernel_initialiser = "he_uniform", activation = "relu", input_fim = 11))
classifier = Sequential()

# Adding the input layer and the first Hidden layer
classifier.add(Dense(units = 6, kernel_initialiser = "he_uniform", activation = "relu", input_fim = 11))
lassifier.add(Dense(units = 6, init = "he_uniform", activation = "relu", input_fim = 11))
classifier.add(Dense(units = 6, init = "he_uniform", activation = "relu", input_fim = 11))
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "relu", input_fim = 11))
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "relu", input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "relu"))
classifier.add(Dense(units = 1, kernel_initializer= = "glorot_uniform", activation = "sigmoid"))
classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))
classifier.summary
classifier.summary()
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Foreplay.py')
model_history = classifier.fit(X_train, Y_train, validation_split = 0.33, batch_size = 10, nb_epoch = 100)
model_history = classifier.fit(X_train, Y_train, validation_split = 0.33, batch_size = 10, epochs = 100)
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)
from sklearn.metrics import accuracy_score
score = accuracy_score(Y_pred, Y_test)
from sklearn.metrics import accuracy_score, confusion_matrix
score = accuracy_score(Y_pred, Y_test)
cm = confusion_matrix(Y_test, Y_pred)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Collection
dataset = pd.read_csv("Churn_Modelling.csv")

# Arranging data
X = dataset.iloc[ : , 3 : 13] # upto 13th column
Y = dataset.iloc[ : , 13]
geography = pd.get_dummies(X["Geography"], drop_first = True)
gender = pd.get_dummies(X["Gender"], drop_first = True)
X = pd.concat([X, geography, gender], axis = 1)
X.drop(["Geography", "Gender"], axis = 1, inplace = True)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Collection
dataset = pd.read_csv("Churn_Modelling.csv")

# Arranging data
X = dataset.iloc[ : , 3 : 13] # upto 13th column
Y = dataset.iloc[ : , 13]

# Handling categorical columns
geography = pd.get_dummies(X["Geography"], drop_first = True)
gender = pd.get_dummies(X["Gender"], drop_first = True)

X = pd.concat([X, geography, gender], axis = 1)
X.drop(["Geography", "Gender"], axis = 1, inplace = True)

# train_tets_split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)

















# # There are 13 features in total
# X = dataset.iloc[ : , 3 : 13]

# # label
# Y = dataset.iloc[ : , 13]

# # In the dataset, geography and gender fields can be
# # used as Groups as they are categorical features, so
# # we should convert them to dummy variables

# geography = pd.get_dummies(X["Geography"], drop_first = True)
# gender = pd.get_dummies(X["Gender"], drop_first = True)

# # Since we have created our dummy fields its time to 
# # concatenate them with the original dataset and drop the
# # categorical columns

# X = pd.concat([X, geography, gender], axis = 1)
# X.drop(['Geography', 'Gender'], axis = 1, inplace = True)

# # Splitting the dataset into training and testing data
# from sklearn.model_selection import train_test_split

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
#                                                     test_size = 0.2,
#                                                     random_state = 42)


# # Feature Scaling
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)


# """Let's make the ANN !!"""
# # import tensorflow as tf
# import keras
# from keras.models import Sequential # This is used to make any type of Neural Network, be it ANN, CNN or RNN, anything
# from keras.layers import Dense # This is used to create hidden layers
# from keras.layers import LeakyReLU, PReLU, ELU
# from keras.layers import Dropout

# # Initialising the NN using Sequential()
# classifier = Sequential()

# """
# 1. units -> output_dim
# 2. kernel_initializer -> initialiser of weights
# 3. activation -> activation function to be used in the respective layer
# """
# # Adding the input layer and the first Hidden layer
# classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", 
#                                  activation = "relu", input_dim = 11)) 
# # input_dim = 11, because we have 11 independent variables in our data

# # Adding the second hidden layer of neurons
# classifier.add(Dense(units = 6, kernel_initializer = "he_uniform",
#                                                  activation = "relu"))

# # Adding the output layer
# classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform", # glorot_uniform is used because it works better in case of ouput layer
#                                                  activation = "sigmoid"))

# """ Compiling the ANN """
# # If our output has only 0 and 1, then Keras documentation tells us
# # to use Binary Cross Entropy, otherwise Categorical Cross Entropy

# classifier.compile(optimizer = "adam", loss = "binary_crossentropy", 
#                                                    metrics = ["accuracy"])


# # Finally fitting our data into the classifier model
# model_history = classifier.fit(X_train, Y_train, validation_split = 0.33,
#                                                batch_size = 10, epochs = 100)

# Y_pred = classifier.predict(X_test)
# Y_pred = (Y_pred > 0.5)

# from sklearn.metrics import accuracy_score, confusion_matrix
# score = accuracy_score(Y_pred, Y_test)
# cm = confusion_matrix(Y_test, Y_pred)
from sklearn.preprocessing import StandardScaler
clf = StandardScaler()
X_train = clf.fit_transform(X_train)
X_test = clf.transform(X_test)
import keras
from keras import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "relu", input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "leaky-relu")
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "relu")
classifier = Sequential()

classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "relu", input_dim = 11))
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "leaky-relu"))
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "leaky_relu"))
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", activation = "relu"))
r = "glorot_uniform", activation = "sigmoid"))

classifier.compile(optimizer = "adam", metrics = ['a
r = "glorot_uniform", activation = "sigmoid"))

classifier.compile(optimizer = "adam", metrics = ['accuracy'], loss = 'binary_croosentropy')
classifer.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))
classifier.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))
classifier.compile(optimizer = "adam", metrics = ['accuracy'], loss = 'binary_croosentropy')
model_history = classifier.fit(X_train, Y_train, epochs = 100, batch_size = 10, validation_split = 0.33)
classifier.compile(optimizer = "adam", metrics = ['accuracy'], loss = 'binary_crossentropy')
model_history = classifier.fit(X_train, Y_train, epochs = 100, batch_size = 10, validation_split = 0.33)
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

from sklearn.metrics import accuracy_score
score = accuracy_score(Y_test, Y_pred)
cm = confusion_matric(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
from sklearn.metrics import accuracy_score, confusion_matrix
score = accuracy_score(Y_test, Y_pred)
cm = confusion_matrix(Y_test, Y_pred)
score = accuracy_score(Y_pred, Y_test)
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Foreplay.py')

## ---(Thu Jul 15 10:02:11 2021)---
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Foreplay.py')
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

classifier = Sequential()
classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", 
                                  activation = "relu", input_dim = 11)) 
# input_dim = 11, because we have 11 independent variables in our data

classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", 
                                  activation = "relu")) 

classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", 
                                  activation = "relu")) 

classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", 
                                  activation = "relu")) 

classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", 
                                  activation = "relu")) 

classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", 
                                  activation = "relu")) 

classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", 
                                  activation = "relu")) 



classifier.add(Dense(units = 6, kernel_initializer = "he_uniform", 
                                  activation = "relu"))

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
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Foreplay.py')
runcell(0, 'C:/Users/Lenovo/.spyder-py3/untitled0.py')
X = data.iloc[ : , 3 : 13]
Y = data.iloc[ : , 13]

geography = pd.get_dummies(X["Geography"], drop_first = True)
gender = pd.get_dummies(X["Gender"], drop_first = True)

X = pd.concat([X, geography, gender], axis = 1)
X.drop(["Geography", " Gender"], axis = 1, inplace = True)
X.drop(["Geography", "Gender"], axis = 1, inplace = True)
geography = pd.get_dummies(X["Geography"], drop_first = True)
gender = pd.get_dummies(X["Gender"], drop_first = True)

X = pd.concat([X, geography, gender], axis = 1)
X.drop(["Geography", "Gender"], axis = 1, inplace = True)

## ---(Thu Jul 15 12:33:53 2021)---
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Hyper_parameter_tuning.py')

## ---(Thu Jul 15 16:47:56 2021)---
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
data = pd.read_csv("Churn_Modelling.csv")
X = data.iloc[ : , 3 : 13]
Y = data.iloc[ : , 13]

# data wrangling
geography = pd.get_dummies(X["Geography"], drop_first = True)
gender = pd.get_dummies(X["Gender"], drop_first = True)

X = pd.concat([X, geography, gender], axis = 1)
X.drop(["Geography", "Gender"], axis = 1, inplace = True)
def create_my_model():
    my_model = Sequential()
    my_model.add(Dense(units = 6, input_dim = 11, activation = "relu"))
    my_model.add(Dense(1, activation = "sigmoid"))
    
    # Compiling the model
    my_model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics = ["accuracy"])
    return my_model

model = KerasClassifier(build_fn = create_my_model)

# Defining the grid search parameters
batchSize = [40, 60, 80, 100]
epochs = [15, 20, 30]

parameter_grid = dict(batch_size = batchSize, epochs = epochs)

my_grid = GridSearchCV(estimator = model, param_grid = parameter_grid, n_jobs = -1, cv = 2)
grid_result = my_grid.fit(X, Y)
print("Best : %f using %s" % (grid_result.best_score_, grid_result.best_params_))
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Hyper_parameter_tuning.py')
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
layers = [[20], [40, 20], [45, 30, 15]]
activations = ["sigmoid", "relu"]
param_grid = dict(layers = layers, activation = activations, batch_size = [10], epochs = [30])
my_grid = GridSearchCV(estimator = model, param_grid = param_grid, cv =5)
print("Best : %f using %s" % (grid_result.best_score_, grid_result.best_params_))
grid_result = my_grid.fit(X, Y)
def create_my_model(layers):
    model = Sequential()
    
    for i, nodes in enumerate(layers):
        if i == 0:
            model.add(Dense(nodes, input_dim = X.shape[1], activation = "relu"))
            
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes, activation = "relu"))
            
            model.add(Dropout(0.3))
    
    model.add(Dense(units = 1, kernel_initializer = "glorot_uniform", activation = "sigmoid"))
    model.compile(optimizer = "adam", metrics = ["accuracy"], loss = "binary_crossentropy")
    
    return model
model = KerasClassifier(build_fn = create_my_model, verbose = 0)

# Defining the grid search parameters
layers = [[20], [40, 20], [45, 30, 15]]
# activations = ["sigmoid", "relu"]


param_grid = dict(layers = layers, batch_size = [10], epochs = [30])

my_grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = 2)
grid_result = my_grid.fit(X, Y)
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
layers = [(20), (40, 20), (45, 30, 15)]
activations = ["sigmoid", "relu"]
param_grid = dict(layers = layers, activation = activations, batch_size = [10], epochs = [30])
my_grid = GridSearchCV(estimator = model, param_grid = param_grid, cv =5)
grid_result = my_grid.fit(X, Y)
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

## ---(Sun Jul 18 03:30:13 2021)---
runcell(0, 'C:/Users/Lenovo/.spyder-py3/untitled0.py')

## ---(Sun Jul 18 10:59:44 2021)---
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Music_Genre_Classification.py')

## ---(Sun Jul 18 21:39:33 2021)---
runcell(0, 'C:/Users/Lenovo/.spyder-py3/untitled0.py')
print(X-test.shape)
print(X_test.shape)
runcell(0, 'C:/Users/Lenovo/.spyder-py3/CNN_Music_genre_classification.py')
print(f"Accuracy on the test set is {test_accuracy}")
runcell(0, 'C:/Users/Lenovo/.spyder-py3/CNN_Music_genre_classification.py')
X_sample = X_test[100]
print(X_sample.shape)
X_sample = X_test.reshape(2497, 13, 130)
print(X_sample.shape)
print(model.predict(X_sample[100]))
print(model.predict(X_sample))
X_sample = X_test[100]
# print(X_sample.shape)

X_sample = X_sample[np.newaxis, ...]
print(X_sample.shape)
runcell(0, 'C:/Users/Lenovo/.spyder-py3/CNN_Music_genre_classification.py')
X_sample = X_test[100]
# print(X_sample.shape)

X_sample = X_sample[np.newaxis, ...]
print(X_sample.shape)
print(model.predict(X_sample))
lis = model.predict(X_sample)
max_num = (max(lis))
max_index = lis.index(max_num)

print(max_index)
print(argmax(model.predict(X_sample)))
print(np.argmax(model.predict(X_sample)))
print(Y_test[100])
print(data["mapping"][6])
data = json.load(DATA_PATH)
print(data["mapping"][6])
with open(data_path, "r") as fp:
    data = json.load(fp)
print(data["mapping"][6])
with open(DATA_PATH, "r") as fp:
    data = json.load(fp)
print(data["mapping"][6])
print(X_test[100])
print(X_test.shape)

## ---(Mon Jul 19 09:59:04 2021)---
from flask import Flask
runcell(0, 'C:/Users/Lenovo/.spyder-py3/untitled0.py')
runcell(0, 'C:/Users/Lenovo/.spyder-py3/Home_Page.py')

## ---(Tue Jul 20 01:26:10 2021)---
runcell(0, 'C:/Users/Lenovo/.spyder-py3/CNN_Music_genre_classification.py')
X_sample = X_test[100]


X_sample = X_sample[np.newaxis, ...]
print(X_sample.shape)


index = (np.argmax(model.predict(X_sample)))

with open(JSON_PATH, "r") as fp:
    data = json.load(fp)
print(data["mapping"][6])
print(X_test.shape)
pickle.dump(model, open('model.pkl'), 'wb')
pickle.dump(model, open('model.pkl', 'wb'))
file_name = 'finalized_model.sav'
pickle.dump(model, open(file_name, 'wb'))
from sklearn.externals import joblib
import sklearn
import joblib
joblib.dump(model, file_name)
model.save('C:\Users\Lenovo\.spyder-py3')
model.save('C:\\Users\\Lenovo\\.spyder-py3')
model = keras.models.load_model('C:\\Users\\Lenovo\\.spyder-py3')
test_error, test_accuracy, = model.evaluate(X_test, Y_test, verbose = 1)
print(f"Accuracy on the test set is {test_accuracy}")
X_sample = X_test[100]


X_sample = X_sample[np.newaxis, ...]
print(X_sample.shape)


index = (np.argmax(model.predict(X_sample)))

with open(JSON_PATH, "r") as fp:
    data = json.load(fp)
print(data["mapping"][6])
print(X_test.shape)