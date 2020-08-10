# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:09:35 2020

@author: sunym
"""

import numpy as np
import tensorflow as tf
import sys 

def INNER(training,model_name):
    X_train = training[:,1:]
    Y_train = training[:,0]
    
    X_trainZ = X_train[:,:2]
    X_trainX = X_train[:,2]
    X_trainX = np.expand_dims(X_trainX,axis=1)
    

    Pain = tf.keras.layers.Input(shape = (1,), dtype=tf.float32)
    WOPain = tf.keras.layers.Input(shape = (2,), dtype = tf.float32)
    
    
    AlphaZ = tf.keras.layers.Dense(200,activation = 'relu',name = 'AlphaZ_1')(WOPain)
    AlphaZ = tf.keras.layers.Dense(100,activation = 'relu',name = 'AlphaZ_2')(AlphaZ)
    A = tf.keras.layers.Dense(1, activation = 'linear',name = 'AlphaZ_3')(AlphaZ)
    
    
    BetaZ = tf.keras.layers.Dense(200,activation = 'relu',name = 'BetaZ_1')(WOPain)
    BetaZ = tf.keras.layers.Dense(100,activation = 'relu',name = 'BetaZ_2')(BetaZ)
    BetaZ = tf.keras.layers.Dense(1, activation = 'linear', name = 'BetaZ_3')(BetaZ)
    
    B = tf.keras.layers.Multiply(name = 'BetaF')([BetaZ,Pain])
    
    y_pred = tf.keras.layers.Add(name = 'model')([A,B])
    y_pred = tf.keras.layers.Activation('sigmoid')(y_pred)
    
    LRmodel = tf.keras.models.Model(inputs=[Pain,WOPain], outputs=y_pred)
    LRmodel.compile(optimizer='sgd',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    history = LRmodel.fit([X_trainX,X_trainZ], Y_train,epochs=100,
                batch_size = 64,verbose=0)
    LRmodel.save(model_name) 
    
if __name__ == "__main__":
    train = np.genfromtxt(sys.argv[1],delimiter=",")
    INNER(train,sys.argv[2])