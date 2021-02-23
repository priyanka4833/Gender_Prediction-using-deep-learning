#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 23:28:00 2020

@author: priyankagore
"""

import pickle
import pandas as pd
import numpy as np
import sys
input = sys.argv[1]
model_path=sys.argv[2]
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

def test_model(input,model_path):
    name=[]
    name.append(input)
    df=pd.DataFrame()
    # Parameters
    predict_col = 'first_name'
    df["first_name"]=name
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
        
    word_vec_length = 15 # Length of the input vector
    char_vec_length = len(accepted_chars) # Length of the character vector
    output_labels = 2 # Number of output labels
    
    print(f"The input vector will have the shape {word_vec_length}x{char_vec_length}.")
    # Define a mapping of chars to integers
    char_to_int = dict((c, i) for i, c in enumerate(accepted_chars))
    print (char_to_int)
    int_to_char = dict((i, c) for i, c in enumerate(accepted_chars))
    print(int_to_char)
    test_x =  np.asarray([np.asarray(name_encoding(char_vec_length,char_to_int,word_vec_length,normalize(accepted_chars,name))) for name in df[predict_col]])
    print(test_x)
    model=pickle.load(open(model_path, 'rb'))
    predict=model.predict(test_x,verbose=0)
    
    predict=["Female" if x>0.5 else "Male" for x in predict]
    predict=''.join(predict)
    print ("The predicted gender is",predict)
    return predict
test_model(input,model_path)