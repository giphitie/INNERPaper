# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 23:44:46 2020

@author: sunym
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn import metrics

Y = pd.read_csv("BaselineNoImputOutcome.csv")
X = pd.read_csv("BaselineNoImputVar.csv")

outcome, count = np.unique(Y, return_counts=True)
prevalence = count[1]/(count[0]+count[1])
X = X.astype(np.float32)
Y = Y.astype(np.float32)

def imput_opioid(dat):
    #BMI
    dat.loc[dat['BMI'].isnull(),'BMI'] = dat.BMI.mean()
    #FMness_d1
    dat.loc[dat['FMness_d1'].isnull(),'FMness_d1'] = dat.FMness_d1.mode().values[0]
    #BPI_PainSeverity2_d1
    dat.loc[dat['BPI_PainSeverity2_d1'].isnull(),'BPI_PainSeverity2_d1'] = \
        dat.BPI_PainSeverity2_d1.mode().values[0]
    #LifeSat_d1
    dat.loc[dat['LifeSat_d1'].isnull(),'LifeSat_d1'] = \
        dat.LifeSat_d1.mode().values[0]
    #depression
    dat.loc[dat['depression'].isnull(),'depression'] = \
        dat.depression.mode().values[0]
    #anxiety
    dat.loc[dat['anxiety'].isnull(),'anxiety'] = \
        dat.anxiety.mode().values[0]
    #anxiety
    dat.loc[dat['body_group'].isnull(),'body_group'] = \
        dat.body_group.mode().values[0]
    #charlson_comorbidity_index
    dat.loc[dat['charlson_comorbidity_index'].isnull(),'charlson_comorbidity_index'] = \
        dat.charlson_comorbidity_index.mode().values[0]
    #alcohol
    dat.loc[dat['alcohol'].isnull(),'alcohol'] = \
        dat.alcohol.mode().values[0]
    #apnea
    dat.loc[dat['apnea'].isnull(),'apnea'] = \
        dat.apnea.mode().values[0]
    #drug
    dat.loc[dat['drug'].isnull(),'drug'] = \
        dat.drug.mode().values[0]
    #smoke
    dat.loc[dat['smoke'].isnull(),'smoke'] = \
        dat.smoke.mode().values[0]
    return dat

def validation(X,Y, TestSize = 0.3, epo = 10,prevalence = 0.5):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TestSize)
    X_train = imput_opioid(X_train)
    X_test = imput_opioid(X_test)
    
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    Y_train = Y_train.to_numpy()
    Y_test = Y_test.to_numpy()
    
    #Balance Sample preparation
    X_train_Opioid = X_train[Y_train[:,0] == 1,:]
    X_train_NoOpioid = X_train[Y_train[:,0] != 1,:]
    Y_train_Opioid = Y_train[Y_train[:,0] == 1,:]
    Y_train_NoOpioid = Y_train[Y_train[:,0] != 1,:]
    
    X_trainZ = np.delete(X_train, 9, 1)
    X_trainX = X_train[:,9]
    X_trainX = np.expand_dims(X_trainX,axis=1)
    
    X_testZ = np.delete(X_test, 9, 1)
    X_testX = X_test[:,9]
    X_testX = np.expand_dims(X_testX,axis=1)
    
    #Build INNER
    Pain = tf.keras.layers.Input(shape = (1,), dtype=tf.float32)
    WOPain = tf.keras.layers.Input(shape = (17,), dtype = tf.float32)
    
    AlphaZ = tf.keras.layers.Dense(250,activation = 'relu',name = 'AlphaZ1')(WOPain)
    AlphaZ = tf.keras.layers.Dense(125,activation = 'relu',name = 'AlphaZ2')(AlphaZ)
    AlphaZ = tf.keras.layers.Dense(64,activation = 'relu',name = 'AlphaZ3')(AlphaZ)
    AlphaZ = tf.keras.layers.Dense(1, activation = 'linear',name = 'AlphaZ5')(AlphaZ)
    
    BetaZ = tf.keras.layers.Dense(250,activation = 'relu',name='BetaZ1')(WOPain)
    BetaZ = tf.keras.layers.Dense(125,activation = 'relu',name='BetaZ2')(BetaZ)
    BetaZ = tf.keras.layers.Dense(64,activation = 'relu',name='BetaZ3')(BetaZ)
    BetaZ = tf.keras.layers.Dense(1, activation = 'linear',name='BetaZ5')(BetaZ)
    
    B = tf.keras.layers.Multiply(name = 'Beta2')([BetaZ,Pain])
    
    y_pred = tf.keras.layers.Add(name = 'model1')([AlphaZ,B])
    y_pred = tf.keras.layers.Activation('sigmoid')(y_pred)
    
    LRmodel = tf.keras.models.Model(inputs=[Pain,WOPain], outputs=y_pred)
    LRmodel.compile(optimizer='sgd',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    
    LRmodel.fit([X_trainX,X_trainZ], Y_train,batch_size =64,epochs=epo,
                      verbose=0)
    _, acc_bench = LRmodel.evaluate([X_testX,X_testZ], Y_test,verbose=0)
    
    #Threshold = 0.5
    predict1prob_bench = LRmodel.predict([X_testX,X_testZ])
    pre_bench = (predict1prob_bench >= 0.5).astype(int)
    tnbench, fpbench, fnbench, tpbench = confusion_matrix(Y_test,pre_bench).ravel()
    sensitivity_bench = tpbench/(tpbench+fnbench)
    specificity_bench = tnbench/(tnbench+fpbench)
    avg_accbench = (sensitivity_bench + specificity_bench)/2
    fpr_bench, tpr_bench, thresholds_bench = metrics.roc_curve(Y_test,predict1prob_bench)
    AUC_bench = metrics.auc(fpr_bench, tpr_bench)
    
    #Threshold = Prevalence
    pre_bench_thre = (predict1prob_bench >= prevalence).astype(int)
    tnbench_thre, fpbench_thre, fnbench_thre, tpbench_thre = confusion_matrix(Y_test,pre_bench_thre).ravel()
    sensitivity_bench_thre = tpbench_thre/(tpbench_thre+fnbench_thre)
    specificity_bench_thre = tnbench_thre/(tnbench_thre+fpbench_thre)
    avg_accbench_thre = (sensitivity_bench_thre + specificity_bench_thre)/2
    acc_bench_thre = (tpbench_thre + tnbench_thre)/(tnbench_thre + fpbench_thre + fnbench_thre + tpbench_thre)    
    del LRmodel
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    
    #Balance sample
    
    pre_balance_all = np.zeros((X_test.shape[0],5))
    for i in range(5):
        idx = np.random.choice(X_train_NoOpioid.shape[0],X_train_Opioid.shape[0],
                               replace = False)
        X_train_NoOpioid1 = X_train_NoOpioid[idx,:]
        Y_train_NoOpioid1 = Y_train_NoOpioid[idx,:]
        X_train_Balance = np.concatenate((X_train_NoOpioid1,X_train_Opioid),axis = 0)
        Y_train_Balance = np.concatenate((Y_train_NoOpioid1,Y_train_Opioid),axis=0)
        shuf_id = np.random.choice(X_train_Balance.shape[0],X_train_Balance.shape[0],replace = False)
        X_train_Balance = X_train_Balance[shuf_id,:]
        Y_train_Balance = Y_train_Balance[shuf_id,:]
        
        X_trainZ_Balance = np.delete(X_train_Balance, 9, 1)
        X_trainX_Balance = X_train_Balance[:,9]
        X_trainX_Balance = np.expand_dims(X_trainX_Balance,axis=1)
        
        #Build INNER
        Pain = tf.keras.layers.Input(shape = (1,), dtype=tf.float32)
        WOPain = tf.keras.layers.Input(shape = (17,), dtype = tf.float32)
        
        AlphaZ = tf.keras.layers.Dense(250,activation = 'relu',name = 'AlphaZ1')(WOPain)
        AlphaZ = tf.keras.layers.Dense(125,activation = 'relu',name = 'AlphaZ2')(AlphaZ)
        AlphaZ = tf.keras.layers.Dense(64,activation = 'relu',name = 'AlphaZ3')(AlphaZ)
        AlphaZ = tf.keras.layers.Dense(1, activation = 'linear',name = 'AlphaZ5')(AlphaZ)
        
        BetaZ = tf.keras.layers.Dense(250,activation = 'relu',name='BetaZ1')(WOPain)
        BetaZ = tf.keras.layers.Dense(125,activation = 'relu',name='BetaZ2')(BetaZ)
        BetaZ = tf.keras.layers.Dense(64,activation = 'relu',name='BetaZ3')(BetaZ)
        BetaZ = tf.keras.layers.Dense(1, activation = 'linear',name='BetaZ5')(BetaZ)
        
        B = tf.keras.layers.Multiply(name = 'Beta2')([BetaZ,Pain])
        
        y_pred = tf.keras.layers.Add(name = 'model1')([AlphaZ,B])
        y_pred = tf.keras.layers.Activation('sigmoid')(y_pred)
        
        LRmodel = tf.keras.models.Model(inputs=[Pain,WOPain], outputs=y_pred)
        LRmodel.compile(optimizer='sgd',
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy'])
        LRmodel.fit([X_trainX_Balance,X_trainZ_Balance], Y_train_Balance,
                    batch_size =64,epochs=epo,verbose=0)
        
        pred_balance = LRmodel.predict([X_testX,X_testZ])
        pre_balance_all[:,i] = pred_balance[:,0]
        del LRmodel
        tf.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        
    
    pre_balance_all = np.sum(pre_balance_all,axis=1)/5
    
    #Balance Sample with threshold=0.5
    pre_bench_thre05_Balance = (pre_balance_all >= 0.5).astype(int)
    tnbench_thre05_Balance, fpbench_thre05_Balance, fnbench_thre05_Balance, tpbench_thre05_Balance = confusion_matrix(Y_test,pre_bench_thre05_Balance).ravel()
    sensitivity_bench_thre05_Balance = tpbench_thre05_Balance/(tpbench_thre05_Balance+fnbench_thre05_Balance)
    specificity_bench_thre05_Balance = tnbench_thre05_Balance/(tnbench_thre05_Balance+fpbench_thre05_Balance)
    avg_accbench_thre05_Balance = (sensitivity_bench_thre05_Balance + specificity_bench_thre05_Balance)/2
    acc_bench_thre05_Balance = (tpbench_thre05_Balance + tnbench_thre05_Balance)/(tnbench_thre05_Balance + fpbench_thre05_Balance + fnbench_thre05_Balance + tpbench_thre05_Balance)    
    fpr_bench_Balance, tpr_bench_Balance, thresholds_bench_Balance = metrics.roc_curve(Y_test,pre_balance_all)
    AUC_bench_Balance = metrics.auc(fpr_bench_Balance, tpr_bench_Balance)
    #Balance Sample with threshold=prevalence
    pre_bench_threPre_Balance = (pre_balance_all >= prevalence).astype(int)
    tnbench_threPre_Balance, fpbench_threPre_Balance, fnbench_threPre_Balance, tpbench_threPre_Balance = confusion_matrix(Y_test,pre_bench_threPre_Balance).ravel()
    sensitivity_bench_threPre_Balance = tpbench_threPre_Balance/(tpbench_threPre_Balance+fnbench_threPre_Balance)
    specificity_bench_threPre_Balance = tnbench_threPre_Balance/(tnbench_threPre_Balance+fpbench_threPre_Balance)
    avg_accbench_threPre_Balance = (sensitivity_bench_threPre_Balance + specificity_bench_threPre_Balance)/2
    acc_bench_threPre_Balance = (tpbench_threPre_Balance + tnbench_threPre_Balance)/(tnbench_threPre_Balance + fpbench_threPre_Balance + fnbench_threPre_Balance + tpbench_threPre_Balance)    
    
    return {"Normal Sample Accuracy":acc_bench,"Normal Sample Balance Accuracy":avg_accbench,"Normal Sample C Stat":AUC_bench,
            "Normal Sample Sensitivity":sensitivity_bench,"Normal Sample Specificity":specificity_bench,
            "Normal Sample Threshold Accuracy":acc_bench_thre,"Normal Sample Threshold Balance Accuracy":avg_accbench_thre,
            "Normal Sample Threshold Sensitivity":sensitivity_bench_thre,
            "Normal Sample Threshold Specificity":specificity_bench_thre,
            "Balance Sample Accuracy":acc_bench_thre05_Balance,"Balance Sample Balance Accuracy":avg_accbench_thre05_Balance,
            "Balance Sample C Stat":AUC_bench_Balance,"Balance Sample Sensitivity":sensitivity_bench_thre05_Balance,
            "Balance Sample Specificity":specificity_bench_thre05_Balance,
            "Balance Sample Threshold Accuracy":acc_bench_threPre_Balance,
            "Balance Sample Threshold Balance Accuracy":avg_accbench_threPre_Balance,
            "Balance Sample Threshold Sensitivity":sensitivity_bench_threPre_Balance,
            "Balance Sample Threshold Specificity":specificity_bench_threPre_Balance,
            "Normal Sample False Positive Rate": fpr_bench,  
            "Normal Sample True Positive Rate": tpr_bench,
            "Normal Sample Threshold":thresholds_bench,
            "Balance Sample False Positive Rate":fpr_bench_Balance,
            "Balance Sample True Positive Rate":tpr_bench_Balance,
            "Balance Sample Threshold":thresholds_bench_Balance}
    
Times = 100
for i in range(0,Times):
    Results = validation(X,Y,epo=200,prevalence = prevalence)
    if i == 0:
        NoSamAcc = Results["Normal Sample Accuracy"]
        NoSamBalAcc = Results["Normal Sample Balance Accuracy"]
        NoSamC = Results["Normal Sample C Stat"]
        NoSamSens = Results["Normal Sample Sensitivity"]
        NoSamSpec = Results["Normal Sample Specificity"]
        NoSamThreAcc = Results["Normal Sample Threshold Accuracy"]
        NoSamThreBalAcc = Results["Normal Sample Threshold Balance Accuracy"]
        NoSamThreSens = Results["Normal Sample Threshold Sensitivity"]
        NoSamThreSpec = Results["Normal Sample Threshold Specificity"]
        NoSamFpr = {i:Results["Normal Sample False Positive Rate"]}
        NoSamTpr = {i:Results["Normal Sample True Positive Rate"]}
        NoSamThre = {i:Results["Normal Sample Threshold"]}
        BaSamAcc = Results["Balance Sample Accuracy"]
        BaSamBalAcc = Results["Balance Sample Balance Accuracy"]
        BaSamC = Results["Balance Sample C Stat"]
        BaSamSens = Results["Balance Sample Sensitivity"]
        BaSamSpec = Results["Balance Sample Specificity"]
        BaSamThreAcc = Results["Balance Sample Threshold Accuracy"]
        BaSamThreBalAcc = Results["Balance Sample Threshold Balance Accuracy"]
        BaSamThreSens = Results["Balance Sample Threshold Sensitivity"]
        BaSamThreSpec = Results["Balance Sample Threshold Specificity"]
        BaSamFpr = {i:Results["Balance Sample False Positive Rate"]}
        BaSamTpr = {i:Results["Balance Sample True Positive Rate"]}
        BaSamThre = {i:Results["Balance Sample Threshold"]}
    else:
        NoSamAcc = np.append(NoSamAcc,Results["Normal Sample Accuracy"])
        NoSamBalAcc = np.append(NoSamBalAcc,Results["Normal Sample Balance Accuracy"])
        NoSamC = np.append(NoSamC,Results["Normal Sample C Stat"])
        NoSamSens = np.append(NoSamSens,Results["Normal Sample Sensitivity"])
        NoSamSpec = np.append(NoSamSpec,Results["Normal Sample Specificity"])
        NoSamThreAcc = np.append(NoSamThreAcc,Results["Normal Sample Threshold Accuracy"])
        NoSamThreBalAcc = np.append(NoSamThreBalAcc,Results["Normal Sample Threshold Balance Accuracy"])
        NoSamThreSens = np.append(NoSamThreSens,Results["Normal Sample Threshold Sensitivity"])
        NoSamThreSpec = np.append(NoSamThreSpec,Results["Normal Sample Threshold Specificity"])
        NoSamFpr.update({i:Results["Normal Sample False Positive Rate"]})
        NoSamTpr.update({i:Results["Normal Sample True Positive Rate"]})
        NoSamThre.update({i:Results["Normal Sample Threshold"]})
        BaSamAcc = np.append(BaSamAcc,Results["Balance Sample Accuracy"])
        BaSamBalAcc = np.append(BaSamBalAcc,Results["Balance Sample Balance Accuracy"])
        BaSamC = np.append(BaSamC,Results["Balance Sample C Stat"])
        BaSamSens = np.append(BaSamSens,Results["Balance Sample Sensitivity"])
        BaSamSpec = np.append(BaSamSpec,Results["Balance Sample Specificity"])
        BaSamThreAcc = np.append(BaSamThreAcc,Results["Balance Sample Threshold Accuracy"])
        BaSamThreBalAcc = np.append(BaSamThreBalAcc,Results["Balance Sample Threshold Balance Accuracy"])
        BaSamThreSens = np.append(BaSamThreSens,Results["Balance Sample Threshold Sensitivity"])
        BaSamThreSpec = np.append(BaSamThreSpec,Results["Balance Sample Threshold Specificity"])
        BaSamFpr.update({i:Results["Balance Sample False Positive Rate"]})
        BaSamTpr.update({i:Results["Balance Sample True Positive Rate"]})
        BaSamThre.update({i:Results["Balance Sample Threshold"]})

        
ModelBasedResults = {"Normal Sample Accuracy":NoSamAcc,"Normal Sample Balance Accuracy":NoSamBalAcc,"Normal Sample C Stat":NoSamC,
            "Normal Sample Sensitivity":NoSamSens,"Normal Sample Specificity":NoSamSpec,
            "Normal Sample Threshold Accuracy":NoSamThreAcc,"Normal Sample Threshold Balance Accuracy":NoSamThreBalAcc,
            "Normal Sample Threshold Sensitivity":NoSamThreSens,
            "Normal Sample Threshold Specificity":NoSamThreSpec,
            "Normal Sample False Positive Rate":NoSamFpr,
            "Normal Sample True Positive Rate":NoSamTpr,
            "Normal Sample Threshold":NoSamThre,
            "Balance Sample Accuracy":BaSamAcc,"Balance Sample Balance Accuracy":BaSamBalAcc,
            "Balance Sample C Stat":BaSamC,"Balance Sample Sensitivity":BaSamSens,
            "Balance Sample Specificity":BaSamSpec,
            "Balance Sample Threshold Accuracy":BaSamThreAcc,
            "Balance Sample Threshold Balance Accuracy":BaSamThreBalAcc,
            "Balance Sample Threshold Sensitivity":BaSamThreSens,
            "Balance Sample Threshold Specificity":BaSamThreSpec,
            "Balance Sample False Positive Rate":BaSamFpr,
            "Balance Sample True Positive Rate":BaSamTpr,
            "Balance Sample Threshold":BaSamThre}
pickle.dump(ModelBasedResults,open("INNER_3Layer_250_v3.p","wb"))
