#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 01:25:51 2020

@author: priyankagore
"""

import json
import pickle
import numpy as np
from flask import Flask, request,render_template
from flask import Flask, redirect, url_for, request
from test_model_flask import test_model
from test_model_flask import *
# 

app = Flask(__name__)

#ML model path
model_path = "/home/priyanka/Downloads/Machine-Learning-Skills-Test/Machine Learning Skills Test/model/lstm_gender_model.sav"

@app.route('/success/<name>')
def success(name):
   return 'The gender is %s' % name

@app.route('/login', methods =["GET", "POST"])
def login():
    if request.method == "POST":
        name = request.form.get("nm")
       
        
    
        #Datapreprocessing Convert the values to float
        
        predict = test_model(name,model_path)
    
        #conf_score =  np.max(classifier.predict_proba([result]))*100
        return redirect(url_for('success',name = predict))
        
   



if __name__ == "__main__":
    app.run()
