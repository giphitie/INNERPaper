# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 18:54:27 2021

@author: sunym
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle

def validation(X,Y, TestSize = 0.2,epo = 30):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TestSize)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    X_trainX = X_train[:,16]
    X_trainZ = np.delete(X_train,16,1)
    X_trainX = np.expand_dims(X_trainX,axis=1)
   
    X_testX = X_test[:,16]
    X_testZ = np.delete(X_test,16,1)
    X_testX = np.expand_dims(X_testX,axis=1)
    
    #build benchmark logistic regression
    X = tf.keras.layers.Input(shape = (1,),dtype = tf.float32)
    Z = tf.keras.layers.Input(shape = (16,),dtype = tf.float32)
    
    Alpha = tf.keras.layers.Dense(1,activation = 'linear',name = 'Alpha')(Z)
    Beta = tf.keras.layers.Dense(1,activation = 'linear',name = 'Beta')(Z)
    BetaF = tf.keras.layers.Multiply(name = 'BetaF')([Beta,X])
    y_pred = tf.keras.layers.Add(name = 'model')([Alpha,BetaF])
    y_pred = tf.keras.activations.sigmoid(y_pred)
    
    Benchmodel = tf.keras.models.Model(inputs=[X,Z], outputs=y_pred)
    Benchmodel.compile(optimizer='sgd',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    history = Benchmodel.fit([X_trainX,X_trainZ], Y_train,epochs=epo,
               batch_size = 64,verbose = 0)
    
    _, acc_bench = Benchmodel.evaluate([X_testX,X_testZ], Y_test,verbose=0)
    predict1prob_bench = Benchmodel.predict([X_testX,X_testZ])
    pre_bench = (predict1prob_bench >= 0.5).astype(int)
    tnbench, fpbench, fnbench, tpbench = confusion_matrix(Y_test,pre_bench).ravel()
    sensitivity_bench = tpbench/(tpbench+fnbench)
    specificity_bench = tnbench/(tnbench+fpbench)
    avg_accbench = (sensitivity_bench + specificity_bench)/2
    fpr_bench, tpr_bench, thresholds_bench = metrics.roc_curve(Y_test,predict1prob_bench)
    AUC_bench = metrics.auc(fpr_bench, tpr_bench)
        
    del Benchmodel
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    return {'Log Accuracy':acc_bench, 'Log C Stat':AUC_bench, 
            'Log Sensitivity':sensitivity_bench,
            'Log Specificity':specificity_bench,'Log Balance Accuracy':avg_accbench}


X = np.genfromtxt("correctModelSR32_x.csv",delimiter=",",skip_header=1)
Y = np.genfromtxt("correctModelSR32_y.csv",delimiter=",",skip_header=1)


Times = 50
logAcc = np.zeros((Times,))
logC = np.zeros((Times,))
logSens = np.zeros((Times,))
logSpec= np.zeros((Times,))
logBalanceAcc = np.zeros((Times,))
for i in range(0,Times):
    Results = validation(X,Y)
    logAcc[i] = Results['Log Accuracy']
    logC[i] = Results['Log C Stat']
    logSens[i] = Results['Log Sensitivity']
    logSpec[i]= Results['Log Specificity']
    logBalanceAcc[i] = Results['Log Balance Accuracy']
    print(i)

Results_all ={'Log Acc':logAcc,'Model Based C Stat': logC,
                 'Model Based Sensitivity':logSens,'Model Based Specificity':logSpec,
                 'Model Based Balance Accuracy':logBalanceAcc}
pickle.dump(Results_all,open("Log_sr32.p","wb"))
