# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:15:59 2020

@author: sunym
"""


import numpy as np
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import sys 

def INNER(training, testing):
    X_train = training[:,1:]
    Y_train = training[:,0]
    X_test = testing[:,1:]
    Y_test = testing[:,0]
    
    X_trainZ = X_train[:,:2]
    X_trainX = X_train[:,2]
    X_trainX = np.expand_dims(X_trainX,axis=1)
    
    X_testZ = X_test[:,:2]
    X_testX = X_test[:,2]
    X_testX = np.expand_dims(X_testX,axis=1)

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
    history = LRmodel.fit([X_trainX,X_trainZ], Y_train,epochs=150,
                batch_size = 64,verbose=0)
    
    _, acc_lr = LRmodel.evaluate([X_testX,X_testZ], Y_test,verbose=0)
    predict1prob_lr = LRmodel.predict([X_testX,X_testZ])
    pre_lr = (predict1prob_lr > 0.5)*1
    pre_lr = np.squeeze(pre_lr)
    tnlr, fplr, fnlr, tplr = confusion_matrix(Y_test,pre_lr).ravel()
    sensitivity_lr = tplr/(tplr+fnlr)
    specificity_lr = tnlr/(tnlr+fplr)
    avg_acclr = (sensitivity_lr + specificity_lr)/2
    fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(Y_test,predict1prob_lr)
    AUC_lr = metrics.auc(fpr_lr, tpr_lr)
    
    return {'C Stat':AUC_lr,'Accuracy':acc_lr,'Sensitivity':sensitivity_lr,
            'Specificity':specificity_lr, 'Balance Accuracy':avg_acclr}

if __name__ == "__main__":
    train = np.genfromtxt(sys.argv[1],delimiter=",")
    test = np.genfromtxt(sys.argv[2],delimiter=",")
    res = INNER(train,test)
    print("C Statistics: ",res['C Stat'],"\n",
          "Accuracy: ",res['Accuracy'],"\n",
          "Sensitivity: ", res['Sensitivity'],"\n",
          "Specificity: ", res['Specificity'],"\n",
          "Balance Accuracy: ", res['Balance Accuracy'],sep="")


