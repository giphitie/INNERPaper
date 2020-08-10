# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:17:01 2020

@author: sunym
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import sys 

def performance_INNER(testing,model_name):
    X_test = testing[:,1:]
    Y_test = testing[:,0]
    
    X_testZ = X_test[:,:2]
    X_testX = X_test[:,2]
    X_testX = np.expand_dims(X_testX,axis=1)
    LRmodel = tf.keras.models.load_model(model_name)
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
    test = np.genfromtxt(sys.argv[1],delimiter=",")
    res = performance_INNER(test,sys.argv[2])
    print("C Statistics: ",res['C Stat'],"\n",
          "Accuracy: ",res['Accuracy'],"\n",
          "Sensitivity: ", res['Sensitivity'],"\n",
          "Specificity: ", res['Specificity'],"\n",
          "Balance Accuracy: ", res['Balance Accuracy'],sep="")