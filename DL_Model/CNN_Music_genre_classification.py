"""
Created on Sun Jul 18 03:30:43 2021

@author: Kartike Sood
"""
import tensorflow as tf
import numpy as np
import librosa
import json
from sklearn.model_selection import train_test_split
import pickle


JSON_PATH = "C:\\Users\\Lenovo\\Deep_Learning\\sth.json"

def load_data(data_path):
    
    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    MFCCs = np.array(data["mfcc"])
    genres = np.array(data["labels"])
    
    return MFCCs, genres

def prepare_datasets(test_size, validation_size):
    # Load the data
    X, Y = load_data(JSON_PATH)
    
    # Train and Test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size)
    
    # Train and Validation sets
    X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size = validation_size)
    
    """
    For CNN, Tensorflow expects a 3D array as input
    => (x_dim, y_dim, num_of_channels[i.e 3 for RGB images, 1 for audio,
        1 for grey scale images, like that]).
    
    Since our training data has no dimension for storing the "Dimension"
    parameter that is desired by our CNN model, we have added a new axis
    to our Training Dataset which will make X_train a 4D object.
                                                                  Now you 
    might be worrying that how can we give it as input to our model as it
    has now become a 4D object and our model requires a 3D one but it is
    not something to worry about because the first dimension of X_train is
    just storing the number of entries in X_train like any other training
    dataset. So when we give X_train as input, only 3 dimensions get passed
    which are :
        1. FIRST
        2. SECOND
        3. THIRD
    
    Even after all this, if I was not able to make you understand, then you can
    watch "VALERIO VELARDO" tutorial on CNN Audio Genre Classification. He might
    be able to help you.
    
    """
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    
    return X_train, X_validation, X_test, Y_train, Y_validation, Y_test

def build_model(input_shape):
    # Create a model
    model = tf.keras.Sequential()
    
    # 1st convolution Layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = input_shape))
    model.add(tf.keras.layers.MaxPool2D((3, 3), strides = (2, 2), padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    
    # 2nd convolution Layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation = "relu", input_shape = input_shape))
    model.add(tf.keras.layers.MaxPool2D((3, 3), strides = (2, 2), padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    
    # 3rd Convolution layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation = "relu", input_shape = input_shape))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides = (2, 2), padding = "same"))
    model.add(tf.keras.layers.BatchNormalization())
    
    # FLatten the output and feed it to dense layer
    model.add(tf.keras.layers.Flatten())
    
    # Adding a Dense Layer
    model.add(tf.keras.layers.Dense(64, activation = "relu"))
    model.add(tf.keras.layers.Dropout(0.3))
    
    # Output Layer
    model.add(tf.keras.layers.Dense(10, activation = "softmax"))
    
    return model

if __name__ == "__main__":
    """ Meta Data Extractor in the house, baby"""
    # meta_data_extractor()
    
    
    # Create train, validation, and test sets
    X_train, X_validation, X_test, Y_train, Y_validation, Y_test = prepare_datasets(test_size = 0.25, validation_size = 0.2)
    
    # Extracting the input shape
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # X_train.shape = (Num_of_segments, 13, estimate_num_MFCCs_per_segment, 1)
    
    # Building the model
    model = build_model(input_shape)
    
    # Compiling the model
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimizer, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    
    # Train the model
    model.fit(X_train, Y_train, validation_data = (X_validation, Y_validation), batch_size = 32, epochs = 30) 
    
    
    # Storing our model
    model.save('C:\\Users\\Lenovo\\.spyder-py3')
    
    # Evaluate the CNN model on test set
    model = tf.keras.models.load_model('C:\\Users\\Lenovo\\.spyder-py3')
    test_error, test_accuracy, = model.evaluate(X_test, Y_test, verbose = 1)
    print(f"Accuracy on the test set is {test_accuracy}")
    
    # Make Prediction for a real life sample
    
    """
    I am sure you will not be able to understand the below written code 
    in first read, but guess what, neither was I. So let me get some
    things straight.
    1. X_test.shape = (2497, 13, 130, 1), i.e 4 dimensions
    2. Our model has been trained by giving the data (of 3 dimensions) of
        MFCC values
    """
    X_sample = X_test[100]
    
    
    X_sample = X_sample[np.newaxis, ...]
    print(X_sample.shape)
    
    
    index = (np.argmax(model.predict(X_sample)))
    
    with open(JSON_PATH, "r") as fp:
        data = json.load(fp)
    print(data["mapping"][6])
    print(X_test.shape)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
