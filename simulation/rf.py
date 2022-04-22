# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 22:52:19 2021

@author: sunym
"""
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.ensemble import RandomForestClassifier

def validation(X,Y, TestSize = 0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TestSize)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    clf_rm = RandomForestClassifier(max_features = "sqrt",min_samples_leaf = 2,
									min_samples_split = 2,n_estimators = 1000)
    clf_rm.fit(X_train, Y_train)
    acc_rm = clf_rm.score(X_test,Y_test)
    pred_prob_rm = clf_rm.predict_proba(X_test)[:,1]
    pred_rm = clf_rm.predict(X_test)
    tn_rm, fp_rm, fn_rm, tp_rm = confusion_matrix(Y_test,pred_rm).ravel()
    sensitivity_rm = tp_rm/(tp_rm+fn_rm)
    specificity_rm = tn_rm/(tn_rm+fp_rm)
    avg_acc_rm= (sensitivity_rm + specificity_rm)/2
    fpr_rm, tpr_rm, thresholds_rm = metrics.roc_curve(Y_test,pred_prob_rm)
    AUC_rm = metrics.auc(fpr_rm, tpr_rm)
    return {'Random Forest Accuracy':acc_rm,'Random Forest C Stat': AUC_rm,
            'Random Forest Sensitivity':sensitivity_rm,
            'Random Forest Specificity':specificity_rm,
            'Random Forest Balance Accuracy':avg_acc_rm}


X = np.genfromtxt("correctModelSR32_x.csv",delimiter=",",skip_header=1)
Y = np.genfromtxt("correctModelSR32_y.csv",delimiter=",",skip_header=1)

Times = 50
RMAccuracy = np.zeros((Times,))
RMC = np.zeros((Times,))
RMSens = np.zeros((Times,))
RMSpec = np.zeros((Times,))
RMBalanceAcc = np.zeros((Times,))
for i in range(0,Times):    
    Results = validation(X,Y)
    RMAccuracy[i] = Results['Random Forest Accuracy']
    RMC[i] = Results['Random Forest C Stat']
    RMSens[i] = Results['Random Forest Sensitivity']
    RMSpec[i] = Results['Random Forest Specificity']
    RMBalanceAcc[i] = Results['Random Forest Balance Accuracy']
    print(i)

Results_all = {'Random Forest Accuracy':RMAccuracy,'Random Forest C Stat': RMC,
            'Random Forest Sensitivity':RMSens,
            'Random Forest Specificity':RMSpec,
            'Random Forest Balance Accuracy':RMBalanceAcc}
pickle.dump(Results_all,open("rf_sr32.p","wb"))
