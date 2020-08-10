# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:27:17 2020

@author: sunym
"""

import numpy as np
import tensorflow as tf
import pandas as pd 
import sys 

def estimate_INNER(data,model_name,file_name):
    X = data[:,1:]
    Y = data[:,0]
    
    X_Z = X[:,:2]
    X_X = X[:,2]
    X_X = np.expand_dims(X_X,axis=1)
    LRmodel = tf.keras.models.load_model(model_name)
    
    input_tensors = [LRmodel.inputs[0], LRmodel.inputs[1],
    ]   
    get_beta = tf.keras.backend.function(input_tensors,
                                      [LRmodel.layers[5].output])
    get_alpha = tf.keras.backend.function(input_tensors, 
                                          [LRmodel.layers[7].output])   
    
    Beta = np.exp(get_beta([X_X,X_Z])[0])
    Alpha= np.exp(get_alpha([X_X,X_Z])[0])
    
    output = np.concatenate((Alpha,Beta),axis=1)
    output_dataframe = pd.DataFrame(data = output,columns=['BOT','POT'])
    output_dataframe.to_csv(file_name,index=False)
    return output_dataframe

if __name__ == "__main__":
     dat = np.genfromtxt(sys.argv[1],delimiter=",")
     res = estimate_INNER(dat,sys.argv[2],sys.argv[3])
     print(res.head())