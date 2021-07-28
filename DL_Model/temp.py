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
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU, PReLU, ELU
from keras.layers import Dropout
