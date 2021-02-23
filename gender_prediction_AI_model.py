#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 22:25:20 2020

@author: priyankagore
"""
#import packages

import pandas as pd
import numpy as np
from numpy import array
from numpy import argmax
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
import sys
import pickle

input_path = sys.argv[1]
output_path = sys.argv[2]


def name_encoding(char_vec_length,char_to_int,word_vec_length,name):

    # Encode input data to int, e.g. a->1, z->26
    integer_encoded = [char_to_int[char] for i, char in enumerate(name) if i < word_vec_length]
    
    # Start one-hot-encoding
    onehot_encoded = list()
    
    for value in integer_encoded:
        # create a list of n zeros, where n is equal to the number of accepted characters
        letter = [0 for _ in range(char_vec_length)]
        letter[value] = 1
        onehot_encoded.append(letter)
        
    # Fill up list to the max length. Lists need do have equal length to be able to convert it into an array
    for _ in range(word_vec_length - len(name)):
        onehot_encoded.append([0 for _ in range(char_vec_length)])
        
    return onehot_encoded

# Removes all non accepted characters
def normalize(accepted_chars,line):
    return [c.lower() for c in line if c.lower() in accepted_chars]

def main(filepath,output_path):
    col_names=['first_name', 'gender',"score"]
    df = (pd.read_csv(filepath, names=col_names))

    # Parameters
    predict_col = 'first_name'
    outout_col = 'gender'
    
    accepted_chars = 'abcdefghijklmnopqrstuvwxyz'
    
    word_vec_length = min(df[predict_col].apply(len).max(), 25) # Length of the input vector
    char_vec_length = len(accepted_chars) # Length of the character vector
    output_labels = 2 # Number of output labels
    
    print(f"The input vector will have the shape {word_vec_length}x{char_vec_length}.")
    # Define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(accepted_chars))
    print (char_to_int)
    int_to_char = dict((i, c) for i, c in enumerate(accepted_chars))
    print(int_to_char)
    # Split dataset in 60% train, 20% test and 20% validation
    train, validate, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])
    #print (train)
    train_y=train["gender"]
    # Convert both the input names as well as the output lables into the discussed machine readable vector format

    train_y=train_y.replace("F",1)
    train_y=train_y.replace("M",0)
    #train_y = label_encoding(train.gender)
    #print(train_y)
    
    
    validate_x = np.asarray([name_encoding(char_vec_length,char_to_int,word_vec_length,normalize(accepted_chars,name)) for name in validate[predict_col]])
    #validate_y = label_encoding(validate.gender)
    validate_y=validate["gender"]
    validate_y=validate_y.replace("F",1)
    validate_y=validate_y.replace("M",0)
    
    test_x = np.asarray([name_encoding(char_vec_length,char_to_int,word_vec_length,normalize(accepted_chars,name)) for name in test[predict_col]])
    #test_y = label_encoding(test.gender)
    test_y=test["gender"]
    test_y=test_y.replace("F",1)
    test_y=test_y.replace("M",0)
    
    hidden_nodes = int(2/3 * (word_vec_length * char_vec_length))
    print(f"The number of hidden nodes is {hidden_nodes}.")
    
    # Build the model
    print('Build model...')
    model = Sequential()
    model.add(LSTM(hidden_nodes, return_sequences=False, input_shape=(word_vec_length, char_vec_length)))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    
    batch_size=1000
    
    train_y = np.array(train_y)
    validate_y = np.array(validate_y)
    class_weights = class_weight.compute_class_weight('balanced',
                                                     np.unique(train_y),
                                                     train_y)
    
    model.fit(train_x, train_y, batch_size=batch_size, epochs=100, validation_data=(validate_x, validate_y), class_weight=class_weights)
    dat=model.predict(test_x,verbose=0)

    dat=[1 if x>0.5 else 0 for x in dat]   
    accuracy=accuracy_score(dat,test_y)
    print("The accuracy of model for gender prediction is ",accuracy)
    
    clf=classification_report(dat,test_y)
    print(clf)   
    filename = 'lstm_gender_model.sav'
    pickle.dump(model, open(output_path+filename, 'wb'))
    
main(input_path,output_path)
    
    
      
        
        


