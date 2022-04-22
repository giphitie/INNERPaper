# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 23:16:15 2021

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
    clf = tree.DecisionTreeClassifier(max_depth = 10,max_features=None,min_samples_leaf = 4,
                                      min_samples_split = 10)
    clf = clf.fit(X_train, Y_train)
    acc_decission_tree = clf.score(X_test,Y_test)
    pred_prob_decission = clf.predict_proba(X_test)[:,1]
    pred_decission = clf.predict(X_test)
    tn_decission, fp_decission, fn_decission, tp_decission = confusion_matrix(Y_test,
                                                                    pred_decission).ravel()
    sensitivity_decission = tp_decission/(tp_decission+fn_decission)
    specificity_decission = tn_decission/(tn_decission+fp_decission)
    avg_acc_decission = (sensitivity_decission + specificity_decission)/2
    fpr_decission, tpr_decission, thresholds_decission = metrics.roc_curve(Y_test,
                                                                           pred_prob_decission)
    AUC_decission = metrics.auc(fpr_decission, tpr_decission)
    
    return {'Decission Tree Accuracy':acc_decission_tree, 'Decission Tree C Stat':AUC_decission, 
            'Decission Tree Sensitivity':sensitivity_decission,
            'Decission Tree Specificity':specificity_decission,
            'Decission Tree Balance Accuracy':avg_acc_decission}


X = np.genfromtxt("incorrectModelSR32_x.csv",delimiter=",",skip_header=1)
Y = np.genfromtxt("incorrectModelSR32_y.csv",delimiter=",",skip_header=1)

Times = 50
DTAcc = np.zeros((Times,))
DTC = np.zeros((Times,))
DTSens = np.zeros((Times,))
DTSpec = np.zeros((Times,))
DTBalanceAcc = np.zeros((Times,))
for i in range(0,Times):    
    Results = validation(X,Y)
    DTAcc[i] = Results['Decission Tree Accuracy']
    DTC[i] = Results['Decission Tree C Stat']
    DTSens[i] = Results['Decission Tree Sensitivity']
    DTSpec[i]= Results['Decission Tree Specificity']
    DTBalanceAcc[i] = Results['Decission Tree Balance Accuracy']
    print(i)

Results_all = {'Decission Tree Accuracy':DTAcc, 'Decission Tree C Stat':DTC, 
            'Decission Tree Sensitivity':DTSens,
            'Decission Tree Specificity':DTSpec,
            'Decission Tree Balance Accuracy':DTBalanceAcc}
pickle.dump(Results_all,open("tree_sr32_rob.p","wb"))