# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 03:30:43 2021

@author: Kartike Sood
"""

import numpy as np
import keras
import matplotlib.pyplot as plt
import librosa
import os
import json
import math

DATASET_PATH = "C:\\Users\\Lenovo\\Desktop\\reduced_data\\genres_original"
JSON_PATH = "C:\\Users\\Lenovo\\Deep_Learning\\sth.json"
SAMPLING_RATE = 22050
DURATION = 30

# json_path is the path of the json file where we want to store all the MFCCs and the labels
# num_segments define the number of segments in which we will divide the song to train our model because we have only 1000 tracks per genre
def save_mfcc(dataset, json_path, n_mfcc = 13, n_fft = 2048,hop_length = 512, num_segments = 5):
    
    ## These variables will be used for the extraction of mfcc values at the end of the loop
    num_samples_per_segment = int((SAMPLING_RATE * DURATION) / num_segments)
    estimated_num_mfcc_per_segment = math.ceil(num_samples_per_segment / hop_length)
    
    # dictionary
    data = {
        "mapping" : [], # genres mapping to list indixes
        "mfcc" : [], # they are the training data
        "labels" : [] # these are the target data
    }
    
    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(DATASET_PATH)): 
        
        # The first iteration will give the path of the whole dataset itself
        # which in turn contains various folders of categories
        if dirpath is not DATASET_PATH:
            
            dirpath_components = os.path.split(dirpath) # ["C", "Users", "Lenovo", "Desktop", "data", "blues"]
            genre = dirpath_components[-1] # "blues"
            data["mapping"].append(genre)
            print(f"Processing {genre}")
            
            for f in filenames:
                
                file_path = os.path.join(dirpath, f)
                signal, sampling_rate = librosa.load(file_path, sr = SAMPLING_RATE)
                
                # process segments, extract MFCC and storing data
                for s in range(num_segments):
                    
                    # If you are not able to understand the below two statements, then you can just imagine a sliding window
                    # which is moving in such a way that it is considering each segment upto "number_of_segments"
                    start_sample = num_samples_per_segment * s 
                    end_sample = num_samples_per_segment * (s + 1) 
                    
                    # This function returns a list of 13 MFCCs which will be used as features in the training of our model
                    mfcc = librosa.feature.mfcc(signal[start_sample : end_sample],
                                                n_mfcc = 13,
                                                n_fft = n_fft,
                                                hop_length = hop_length)
                    
                    if estimated_num_mfcc_per_segment == mfcc.shape[1]: # This condition ensures that if any audio clip has different number of MFCCs than estimated, then that audio clip will not be considered
                        data["mfcc"].append(mfcc.tolist())
                        data["labels"].append(i - 1) # Here "i - 1" is written because the first iteration of os.walk() function will give us the directory of the dataset which we will avoid
                        
                        print(f"{file_path} with segment {s + 1}")
                        
    with open(json_path, "w") as fp:
         json.dump(data, fp, indent = 4)
            
if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments = 10)