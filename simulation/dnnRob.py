# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 18:43:07 2021

@author: sunym
"""
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle

def validation(X,Y, TestSize = 0.2,epo = 30):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    #build the regular nn
    Input = tf.keras.layers.Input(shape = (17,), dtype = tf.float32)
    
    X = tf.keras.layers.Dense(100,activation = 'relu')(Input)
    X = tf.keras.layers.Dropout(rate = 0.0)(X)
    X = tf.keras.layers.Dense(100,activation = 'relu')(X)
    X = tf.keras.layers.Dropout(rate = 0.0)(X)
    X = tf.keras.layers.Dense(160,activation = 'relu')(X)
    X = tf.keras.layers.Dropout(rate = 0.3)(X)
    y_pred = tf.keras.layers.Dense(1, activation = 'sigmoid')(X)
    
    REmodel = tf.keras.models.Model(inputs=[Input], outputs=y_pred)
    REmodel.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00046),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    history = REmodel.fit(X_train, Y_train,validation_data = (X_test,Y_test),
                          epochs=epo,batch_size = 64,verbose = 1)
    
    _, acc_re = REmodel.evaluate(X_test, Y_test,verbose = 0)
    pre_re = REmodel.predict(X_test)
    predict1prob_re = pre_re[:,0]
    pre_re = (pre_re >=0.5).astype(int)
    tnre, fpre, fnre, tpre = confusion_matrix(Y_test,pre_re).ravel()
    sensitivity_re = tpre/(tpre+fnre)
    specificity_re = tnre/(tnre+fpre)
    avg_accre = (sensitivity_re + specificity_re)/2
    
    fpr_re, tpr_re, thresholds_re = metrics.roc_curve(Y_test,predict1prob_re)
    AUC_re = metrics.auc(fpr_re, tpr_re)
    
    del REmodel
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    return {'DNN Accuracy':acc_re, 'DNN C Stat':AUC_re, 
            'DNN Sensitivity':sensitivity_re,
            'DNN Specificity':specificity_re,'DNN Balance Accuracy':avg_accre}


X = np.genfromtxt("incorrectModelSR32_x.csv",delimiter=",",skip_header=1)
Y = np.genfromtxt("incorrectModelSR32_y.csv",delimiter=",",skip_header=1)


Times = 50
DNNAcc = np.zeros((Times,))
DNNC = np.zeros((Times,))
DNNSens = np.zeros((Times,))
DNNSpec= np.zeros((Times,))
DNNBalanceAcc = np.zeros((Times,))
for i in range(0,Times):
    Results = validation(X,Y,epo = 60)
    DNNAcc[i] = Results['DNN Accuracy']
    DNNC[i] = Results['DNN C Stat']
    DNNSens[i] = Results['DNN Sensitivity']
    DNNSpec[i]= Results['DNN Specificity']
    DNNBalanceAcc[i] = Results['DNN Balance Accuracy']
    print(i)

Results_all = {'DNN Acc':DNNAcc,'DNN C Stat': DNNC,
                 'DNN Sensitivity':DNNSens,'DNN Specificity':DNNSpec,
                 'DNN Balance Accuracy':DNNBalanceAcc}
pickle.dump(Results_all,open("DNN_sr32_rob.p","wb"))
