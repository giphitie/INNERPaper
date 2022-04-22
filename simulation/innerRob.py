# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 12:53:44 2019

@author: sunym
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle

def validation(X,Y, TestSize = 0.2,epo = 60):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    X_trainX = X_train[:,16]
    X_trainZ = np.delete(X_train,16,1)
    X_trainX = np.expand_dims(X_trainX,axis=1)
    
    X_testX = X_test[:,16]
    X_testZ = np.delete(X_test,16,1)
    X_testX = np.expand_dims(X_testX,axis=1)
    
    
    #build the model based nn
    Pain = tf.keras.layers.Input(shape = (1,), dtype=tf.float32)
    WOPain = tf.keras.layers.Input(shape = (16,), dtype = tf.float32)
    
    AlphaZ = tf.keras.layers.Dense(200,activation = 'relu',name = 'AlphaZ_1')(WOPain)
    AlphaZ = tf.keras.layers.Dropout(rate = 0.0,name = 'AlphaZ_Dropout_1')(AlphaZ)
    AlphaZ = tf.keras.layers.Dense(10,activation = 'relu',name = 'AlphaZ_2')(AlphaZ)
    AlphaZ = tf.keras.layers.Dropout(rate = 0.0,name = 'AlphaZ_Dropout_2')(AlphaZ)
    A = tf.keras.layers.Dense(1, activation = 'linear',name = 'AlphaZ_3')(AlphaZ)
    
    BetaZ = tf.keras.layers.Dense(180,activation = 'relu',name = 'BetaZ_1')(WOPain)
    BetaZ = tf.keras.layers.Dropout(rate = 0,name = 'BetaZ_Dropout_1')(BetaZ)
    BetaZ = tf.keras.layers.Dense(90,activation = 'relu',name = 'BetaZ_2')(BetaZ)
    BetaZ = tf.keras.layers.Dropout(rate = 0,name = 'BetaZ_Dropout_2')(BetaZ)
    BetaZ = tf.keras.layers.Dense(1, activation = 'linear', name = 'BetaZ_3')(BetaZ)
    
    
    B = tf.keras.layers.Multiply(name = 'BetaF')([BetaZ,Pain])
    
    y_pred = tf.keras.layers.Add(name = 'model')([A,B])
    y_pred = tf.keras.layers.Activation('sigmoid')(y_pred)
    
    
    LRmodel = tf.keras.models.Model(inputs=[Pain,WOPain], outputs=y_pred)
    LRmodel.compile(
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.004),
        metrics=["accuracy"],
    )
    
    history = LRmodel.fit([X_trainX,X_trainZ], Y_train,epochs=epo,
                          validation_data = ([X_testX,X_testZ],Y_test),
                          batch_size = 64,verbose = 1)
    
    _, acc_lr = LRmodel.evaluate([X_testX,X_testZ], Y_test,verbose=0)
    pre_lr = LRmodel.predict([X_testX,X_testZ])
    predict1prob_lr = pre_lr[:,0]
    pre_lr = (pre_lr > 0.5).astype(int)
    tnlr, fplr, fnlr, tplr = confusion_matrix(Y_test,pre_lr).ravel()
    sensitivity_lr = tplr/(tplr+fnlr)
    specificity_lr = tnlr/(tnlr+fplr)
    avg_acclr = (sensitivity_lr + specificity_lr)/2
    fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(Y_test,predict1prob_lr)
    AUC_lr = metrics.auc(fpr_lr, tpr_lr)
    
    del LRmodel
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    return {'Model Based Accuracy':acc_lr, 'Model Based C Stat':AUC_lr, 
            'Model Based Sensitivity':sensitivity_lr,
            'Model Based Specificity':specificity_lr,'Model Based Balance Accuracy':avg_acclr}


X = np.genfromtxt("incorrectModelSR32_x.csv",delimiter=",",skip_header=1)
Y = np.genfromtxt("incorrectModelSR32_y.csv",delimiter=",",skip_header=1)



Times = 50
ModelBasedAcc = np.zeros((Times,))
ModelBasedC = np.zeros((Times,))
ModelBasedSens = np.zeros((Times,))
ModelBasedSpec= np.zeros((Times,))
ModelBasedBalanceAcc = np.zeros((Times,))
for i in range(0,Times):
    Results = validation(X,Y)
    ModelBasedAcc[i] = Results['Model Based Accuracy']
    ModelBasedC[i] = Results['Model Based C Stat']
    ModelBasedSens[i] = Results['Model Based Sensitivity']
    ModelBasedSpec[i]= Results['Model Based Specificity']
    ModelBasedBalanceAcc[i] = Results['Model Based Balance Accuracy']
    print(i)

Results_all = {'Model Based Acc':ModelBasedAcc,'Model Based C Stat': ModelBasedC,
                 'Model Based Sensitivity':ModelBasedSens,'Model Based Specificity':ModelBasedSpec,
                 'Model Based Balance Accuracy':ModelBasedBalanceAcc}
pickle.dump(Results_all,open("INNER_sr32_rob.p","wb"))
