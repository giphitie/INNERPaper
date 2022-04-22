# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 17:35:03 2021

@author: sunym
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.svm import SVC

def validation(X,Y, TestSize = 0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    
    clf_svc = SVC(gamma = 0.1,kernel='rbf',C = 10)
    clf_svc.fit(X_train, Y_train)
    acc_svc = clf_svc.score(X_test,Y_test)
    pred_prob_svc = clf_svc.decision_function(X_test)
    pred_svc = clf_svc.predict(X_test)
    tn_svc, fp_svc, fn_svc, tp_svc = confusion_matrix(Y_test,pred_svc).ravel()
    sensitivity_svc = tp_svc/(tp_svc+fn_svc)
    specificity_svc = tn_svc/(tn_svc+fp_svc)
    avg_acc_svc= (sensitivity_svc + specificity_svc)/2
    fpr_svc, tpr_svc, thresholds_svc = metrics.roc_curve(Y_test,pred_prob_svc)
    AUC_svc = metrics.auc(fpr_svc, tpr_svc)
    return {'SVM Accuracy':acc_svc, 'SVM C Stat':AUC_svc, 
            'SVM Sensitivity':sensitivity_svc,
            'SVM Specificity':specificity_svc,
            'SVM Balance Accuracy':avg_acc_svc}

X = np.genfromtxt("correctModelSR32_x.csv",delimiter=",",skip_header=1)
Y = np.genfromtxt("correctModelSR32_y.csv",delimiter=",",skip_header=1)


Times = 50
svcAcc = np.zeros((Times,))
svcC = np.zeros((Times,))
svcSens = np.zeros((Times,))
svcSpec = np.zeros((Times,))
svcBalanceAcc = np.zeros((Times,))
for i in range(0,Times):    
    Results = validation(X,Y)
    svcAcc[i] = Results['SVM Accuracy']
    svcC[i] = Results['SVM C Stat']
    svcSens[i] = Results['SVM Sensitivity']
    svcSpec[i]= Results['SVM Specificity']
    svcBalanceAcc[i] = Results['SVM Balance Accuracy']
    print(i)

Results_all = {'SVM Accuracy':svcAcc, 'SVM C Stat':svcC, 
            'SVM Sensitivity':svcSens,
            'SVM Specificity':svcSpec,
            'SVM Balance Accuracy':svcBalanceAcc}
pickle.dump(Results_all,open("svm_sr32.p","wb"))
