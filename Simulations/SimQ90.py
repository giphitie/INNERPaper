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

def validation(X,Y, TestSize = 0.2, epo = 10):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TestSize)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_train_cat = tf.keras.utils.to_categorical(Y_train, 2)
    Y_test_cat = tf.keras.utils.to_categorical(Y_test, 2)
    X_trainZ = X_train[:,:2]
    X_trainX = X_train[:,2]
    X_trainX = np.expand_dims(X_trainX,axis=1)
    X_trainOnesA = np.ones((X_trainX.shape[0],1),dtype = np.float32)
    X_trainOnesB = np.ones((X_trainX.shape[0],1),dtype = np.float32)
    X_testZ = X_test[:,:2]
    X_testX = X_test[:,2]
    X_testX = np.expand_dims(X_testX,axis=1)
    X_testOnesA = np.ones((X_testX.shape[0],1),dtype = np.float32)
    X_testOnesB = np.ones((X_testX.shape[0],1),dtype = np.float32)
    trueLabel = np.argmax(Y_test_cat,axis = 1)
    
    #build benchmark logistic regression
    X = tf.keras.layers.Input(shape = (1,),dtype = tf.float32)
    Z = tf.keras.layers.Input(shape = (2,),dtype = tf.float32)
    
    Alpha = tf.keras.layers.Dense(1,activation = 'linear',name = 'Alpha')(Z)
    Beta = tf.keras.layers.Dense(1,activation = 'linear',name = 'Beta')(Z)
    BetaF = tf.keras.layers.Multiply(name = 'BetaF')([Beta,X])
    y_pred = tf.keras.layers.Add(name = 'model')([Alpha,BetaF])
    y_pred = tf.keras.activations.sigmoid(y_pred)
    
    Benchmodel = tf.keras.models.Model(inputs=[X,Z], outputs=y_pred)
    Benchmodel.compile(optimizer='sgd',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    Benchmodel.fit([X_trainX,X_trainZ], Y_train,epochs=epo,
               batch_size = 64,verbose=0)
    
    _, acc_bench = Benchmodel.evaluate([X_testX,X_testZ], Y_test,verbose=0)
    predict1prob_bench = Benchmodel.predict([X_testX,X_testZ])
    pre_bench = (predict1prob_bench >= 0.5).astype(int)
    tnbench, fpbench, fnbench, tpbench = confusion_matrix(Y_test,pre_bench).ravel()
    sensitivity_bench = tpbench/(tpbench+fnbench)
    specificity_bench = tnbench/(tnbench+fpbench)
    avg_accbench = (sensitivity_bench + specificity_bench)/2
    fpr_bench, tpr_bench, thresholds_bench = metrics.roc_curve(Y_test,predict1prob_bench)
    AUC_bench = metrics.auc(fpr_bench, tpr_bench)
    
    tf.keras.backend.clear_session()
    
    #build the model based nn
    Pain = tf.keras.layers.Input(shape = (1,), dtype=tf.float32)
    WOPain = tf.keras.layers.Input(shape = (2,), dtype = tf.float32)
    #OnesA = tf.keras.layers.Input(shape = (1,), dtype = tf.float32)
    #OnesB = tf.keras.layers.Input(shape = (1,), dtype = tf.float32)
    
    #Alpha0 = tf.keras.layers.Dense(1, activation = 'linear',name='Alpha0', use_bias=False)(OnesA)
    
    AlphaZ = tf.keras.layers.Dense(200,activation = 'relu',name = 'AlphaZ_1')(WOPain)
    #AlphaZ = tf.keras.layers.Dropout(rate = 0.25,name = 'AlphaZ_Dropout_1')(AlphaZ)
    AlphaZ = tf.keras.layers.Dense(100,activation = 'relu',name = 'AlphaZ_2')(AlphaZ)
    #AlphaZ = tf.keras.layers.Dropout(rate = 0.25,name = 'AlphaZ_Dropout_2')(AlphaZ)
    A = tf.keras.layers.Dense(1, activation = 'linear',name = 'AlphaZ_3')(AlphaZ)
    
    #A = tf.keras.layers.Add(name = 'Alpha')([AlphaZ, Alpha0])
    
    #Beta0 = tf.keras.layers.Dense(1, activation = 'linear',name = 'Beta0', use_bias=False)(OnesB)
    
    BetaZ = tf.keras.layers.Dense(200,activation = 'relu',name = 'BetaZ_1')(WOPain)
    #BetaZ = tf.keras.layers.Dropout(rate = 0.25,name = 'BetaZ_Dropout_1')(BetaZ)
    BetaZ = tf.keras.layers.Dense(100,activation = 'relu',name = 'BetaZ_2')(BetaZ)
    #BetaZ = tf.keras.layers.Dropout(rate = 0.25,name = 'BetaZ_Dropout_2')(BetaZ)
    BetaZ = tf.keras.layers.Dense(1, activation = 'linear', name = 'BetaZ_3')(BetaZ)
    
    #BetaZ = tf.keras.layers.Add(name = 'Beta')([BetaZ,Beta0])
    B = tf.keras.layers.Multiply(name = 'BetaF')([BetaZ,Pain])
    
    y_pred = tf.keras.layers.Add(name = 'model')([A,B])
    y_pred = tf.keras.layers.Dense(2, activation='softmax')(y_pred)
    
    LRmodel = tf.keras.models.Model(inputs=[Pain,WOPain], outputs=y_pred)
    LRmodel.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    LRmodel.fit([X_trainX,X_trainZ], Y_train_cat,epochs=epo,
                batch_size = 64,verbose=0)
    LRmodel.fit([X_trainX,X_trainZ], Y_train_cat,epochs=int(epo/2),batch_size=X_train.shape[0],verbose=0)
    
    _, acc_lr = LRmodel.evaluate([X_testX,X_testZ], Y_test_cat,verbose=0)
    pre_lr = LRmodel.predict([X_testX,X_testZ])
    predict1prob_lr = pre_lr[:,1]
    pre_lr = np.argmax(pre_lr,axis = 1)
    tnlr, fplr, fnlr, tplr = confusion_matrix(trueLabel,pre_lr).ravel()
    sensitivity_lr = tplr/(tplr+fnlr)
    specificity_lr = tnlr/(tnlr+fplr)
    avg_acclr = (sensitivity_lr + specificity_lr)/2
    fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(trueLabel,predict1prob_lr)
    AUC_lr = metrics.auc(fpr_lr, tpr_lr)
    
    tf.keras.backend.clear_session()
    
    #build the regular nn
    Input = tf.keras.layers.Input(shape = (3,), dtype = tf.float32)
    
    X = tf.keras.layers.Dense(200,activation = 'relu')(Input)
    X = tf.keras.layers.Dropout(rate = 0.25)(X)
    X = tf.keras.layers.Dense(200,activation = 'relu')(X)
    X = tf.keras.layers.Dropout(rate = 0.25)(X)
    X = tf.keras.layers.Dense(200,activation = 'relu')(X)
    X = tf.keras.layers.Dropout(rate = 0.25)(X)
    X = tf.keras.layers.Dense(1, activation = 'linear')(X)
    
    y_pred = tf.keras.layers.Dense(2, activation='softmax')(X)
    
    REmodel = tf.keras.models.Model(inputs=[Input], outputs=y_pred)
    REmodel.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    REmodel.fit(X_train, Y_train_cat,epochs=epo,batch_size = 64,verbose=0)
    
   
    _, acc_re = REmodel.evaluate(X_test, Y_test_cat,verbose = 0)
    pre_re = REmodel.predict(X_test)
    predict1prob_re = pre_re[:,1]
    pre_re = np.argmax(pre_re,axis=1)
    tnre, fpre, fnre, tpre = confusion_matrix(trueLabel,pre_re).ravel()
    sensitivity_re = tpre/(tpre+fnre)
    specificity_re = tnre/(tnre+fpre)
    avg_accre = (sensitivity_re + specificity_re)/2
    
    fpr_re, tpr_re, thresholds_re = metrics.roc_curve(trueLabel,predict1prob_re)
    AUC_re = metrics.auc(fpr_re, tpr_re)
    tf.keras.backend.clear_session()
    
    
    #tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(trueLabel, predictions_lr).ravel()
    #sensitivity_lr = tp_lr/(tp_lr+fn_lr)
    #specificity_lr = tn_lr/(tn_lr+fp_lr)
    #avg_acc_lr = (sensitivity_lr + specificity_lr)/2
    #tn_re, fp_re, fn_re, tp_re = confusion_matrix(trueLabel, predictions_re).ravel()
    #sensitivity_re = tp_re/(tp_re+fn_re)
    #specificity_re = tn_re/(tn_re+fp_re)
    #avg_acc_re = (sensitivity_re + specificity_re)/2
    
    return {'Model Based Accuracy':acc_lr, 'Model Based C Stat':AUC_lr, 
            'RNN Accuracy':acc_re,'RNN C Stat':AUC_re,'Model Based Sensitivity':sensitivity_lr,
            'Model Based Specificity':specificity_lr,'Model Based Balance Accuracy':avg_acclr,
            'RNN Sensitivity':sensitivity_re,'RNN Specificity':specificity_re,
            'RNN Balance Accuracy':avg_accre,
            'Logistic Accuracy':acc_bench,'Logistic C Stat': AUC_bench,
            'Logistic Sensitivity':sensitivity_bench,
            'Logistic Specificity':specificity_bench,
            'Logistic Balance Accuracy':avg_accbench}


Y = np.genfromtxt("SimData_Y_q90_20K.csv",delimiter=",",skip_header=1)
X = np.genfromtxt("SimData_X_q90_20K.csv",delimiter=",",skip_header=1)

Times = 500
for i in range(0,Times):
    Results = validation(X,Y,epo=100)
    if i == 0:
        ModelBasedAcc = Results['Model Based Accuracy']
        ModelBasedC = Results['Model Based C Stat']
        ModelBasedSens = Results['Model Based Sensitivity']
        ModelBasedSpec= Results['Model Based Specificity']
        ModelBasedBalanceAcc = Results['Model Based Balance Accuracy']
        RNNAccuracy = Results['RNN Accuracy']
        RNNC = Results['RNN C Stat']
        RNNSens = Results['RNN Sensitivity']
        RNNSpec = Results['RNN Specificity']
        RNNBalanceAcc = Results['RNN Balance Accuracy']
        LogAccuracy = Results['Logistic Accuracy']
        LogC = Results['Logistic C Stat']
        LogSens = Results['Logistic Sensitivity']
        LogSpec = Results['Logistic Specificity']
        LogBalanceAcc = Results['Logistic Balance Accuracy']
    else:
        ModelBasedAcc = np.append(ModelBasedAcc,Results['Model Based Accuracy'])
        ModelBasedC = np.append(ModelBasedC,Results['Model Based C Stat'])
        ModelBasedSens = np.append(ModelBasedSens,Results['Model Based Sensitivity'])
        ModelBasedSpec = np.append(ModelBasedSpec,Results['Model Based Specificity'])
        ModelBasedBalanceAcc = np.append(ModelBasedBalanceAcc,Results['Model Based Balance Accuracy'])
        RNNAccuracy = np.append(RNNAccuracy,Results['RNN Accuracy'])
        RNNC = np.append(RNNC,Results['RNN C Stat'])
        RNNSens = np.append(RNNSens,Results['RNN Sensitivity'])
        RNNSpec = np.append(RNNSpec,Results['RNN Specificity'])
        RNNBalanceAcc = np.append(RNNBalanceAcc,Results['RNN Balance Accuracy'])
        LogAccuracy = np.append(LogAccuracy,Results['Logistic Accuracy'])
        LogC = np.append(LogC,Results['Logistic C Stat'])
        LogSens = np.append(LogSens,Results['Logistic Sensitivity'])
        LogSpec = np.append(LogSpec,Results['Logistic Specificity'])
        LogBalanceAcc = np.append(LogBalanceAcc,Results['Logistic Balance Accuracy'])
    print(i)

SimQ90Results = {'Model Based Acc':ModelBasedAcc,'Model Based C Stat': ModelBasedC,
                 'RNN Accuracy': RNNAccuracy,'RNN C Stat': RNNC,
                 'Model Based Sensitivity':ModelBasedSens,'Model Based Specificity':ModelBasedSpec,
                 'Model Based Balance Accuracy':ModelBasedBalanceAcc,'RNN Sensitivity':RNNSens,
                 'RNN Specificity':RNNSpec,'RNN Balance Accuracy':RNNBalanceAcc,
                 'Logistic Accuracy':LogAccuracy,'Logistic C Stat': LogC,
                'Logistic Sensitivity':LogSens,
                'Logistic Specificity':LogSpec,
                'Logistic Balance Accuracy':LogBalanceAcc}
pickle.dump(SimQ90Results,open("SimQ90Results_500_iter.p","wb"))
