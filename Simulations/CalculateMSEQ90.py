# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:55:56 2019

@author: sunym
"""

import numpy as np
import tensorflow as tf
#from sklearn.model_selection import train_test_split
#from sklearn import metrics
#from sklearn.metrics import confusion_matrix
import pickle

z1 = np.linspace(-12, 0, 100)
z2 = np.linspace(0, 12, 100)
z_all = np.array([(x,y) for x in z1 for y in z2])

#Correct Model
def true_beta (z1, z2):
    tb = 0.35 + np.sin(0.75 * z1) - np.cos(0.75 * z2)
    return np.expand_dims(tb,axis = 1)

def calcualte_mse (TrueAlpha, PredictAlpha):
    return np.mean((TrueAlpha - PredictAlpha)**2)

def validation(X,Y,InputZArray, Initial_weights = None,epo = 10):
    X = X.astype(np.float32)
    Y_cat = tf.keras.utils.to_categorical(Y, 2)
    #Y_test_cat = tf.keras.utils.to_categorical(Y_test, 2)
    X_Z = X[:,:2]
    X_X = X[:,2]
    X_X = np.expand_dims(X_X,axis=1)
    #X_OnesA = np.ones((X_X.shape[0],1),dtype = np.float32)
    #X_OnesB = np.ones((X_X.shape[0],1),dtype = np.float32)
    #X_testZ = X_test[:,:2]
    #X_testX = X_test[:,2]
    #X_testX = np.expand_dims(X_testX,axis=1)
    #X_testOnesA = np.ones((X_testX.shape[0],1),dtype = np.float32)
    #X_testOnesB = np.ones((X_testX.shape[0],1),dtype = np.float32)
    
    tf.compat.v1.disable_eager_execution()    
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
    
    if not Initial_weights is None:
        LRmodel.set_weights(Initial_weights)
    #predict beta
    Inputs = [LRmodel.inputs[1],
              tf.keras.backend.learning_phase()
    ]
    get_beta1 = tf.keras.backend.function(Inputs,
                                      [LRmodel.layers[5].output])
    
    LRmodel.fit([X_X,X_Z], Y_cat,epochs=epo,batch_size=64, verbose=0)
    LRmodel.fit([X_X,X_Z], Y_cat,epochs=int(epo/2),batch_size=X.shape[0], verbose=0)
    #history = LRmodel.fit([X_X,X_Z,X_OnesA,X_OnesB], Y_cat,epochs=100,verbose=0,validation_split=0.3)

    SoftmaxWeights = LRmodel.layers[10].get_weights()[0]
    Delta = SoftmaxWeights[0,1] - SoftmaxWeights[0,0]
    
    InputZArray = InputZArray.astype(np.float32)
    InputOnes = np.ones((InputZArray.shape[0],1),dtype = np.float32)


    PredictBeta = get_beta1([InputZArray,0])
    PredictBeta = PredictBeta[0]*Delta
    weights = LRmodel.get_weights()
    tf.keras.backend.clear_session()
    
    #TrueBeta = true_beta(InputZArray[:,0],InputZArray[:,1])
    #mse = calcualte_mse(TrueBeta,PredictBeta)
    
    #tn_lr, fp_lr, fn_lr, tp_lr = confusion_matrix(trueLabel, predictions_lr).ravel()
    #sensitivity_lr = tp_lr/(tp_lr+fn_lr)
    #specificity_lr = tn_lr/(tn_lr+fp_lr)
    #avg_acc_lr = (sensitivity_lr + specificity_lr)/2
    #tn_re, fp_re, fn_re, tp_re = confusion_matrix(trueLabel, predictions_re).ravel()
    #sensitivity_re = tp_re/(tp_re+fn_re)
    #specificity_re = tn_re/(tn_re+fp_re)
    #avg_acc_re = (sensitivity_re + specificity_re)/2
    
    return {"Beta":PredictBeta,"Weights":weights}

TB = true_beta(z_all[:,0],z_all[:,1])

Y = np.genfromtxt("SimData_Y_q90_40K.csv",delimiter=",",skip_header=1)
X = np.genfromtxt("SimData_X_q90_40K.csv",delimiter=",",skip_header=1)

MSE = {"MSE_S3.5K":np.zeros((100,)),"MSE_S7K":np.zeros((100,)),
       "MSE_S15K":np.zeros((100,))}

Times = 100
for i in range(0,Times):
    shuf_ind = np.random.choice(range(X.shape[0]),size = X.shape[0],replace = False,
                            p = None)
    X = X[shuf_ind,:]
    Y = Y[shuf_ind,]
    results10K = validation(X[:3500,:],Y[:3500,],z_all,epo=100)
    results20K = validation(X[:7000,:],Y[:7000,],z_all,Initial_weights = 
                            results10K["Weights"],epo = 50)
    results40K = validation(X[:15000,:],Y[:15000,],z_all,Initial_weights = results20K["Weights"],
                            epo = 100)
    MSE["MSE_S3.5K"][i] = calcualte_mse(TB,results10K["Beta"])
    MSE["MSE_S7K"][i] = calcualte_mse(TB,results20K["Beta"])
    MSE["MSE_S15K"][i] = calcualte_mse(TB,results40K["Beta"])
    #if i == 29:
     #   pickle.dump(MSE,open("MSEQ90_30.p","wb"))    

pickle.dump(MSE,open("MSEQ90S3.5K.p","wb"))



