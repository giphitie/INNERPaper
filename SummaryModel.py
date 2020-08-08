# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 01:18:43 2020

@author: sunym
"""

import tensorflow as tf

Pain = tf.keras.layers.Input(shape = (1,), dtype=tf.float32)
WOPain = tf.keras.layers.Input(shape = (17,), dtype = tf.float32)

AlphaZ = tf.keras.layers.Dense(250,activation = 'relu',name = 'AlphaZ1')(WOPain)
AlphaZ = tf.keras.layers.Dense(125,activation = 'relu',name = 'AlphaZ2')(AlphaZ)
AlphaZ = tf.keras.layers.Dense(1, activation = 'linear',name = 'AlphaZ5')(AlphaZ)

BetaZ = tf.keras.layers.Dense(250,activation = 'relu',name='BetaZ1')(WOPain)
BetaZ = tf.keras.layers.Dense(125,activation = 'relu',name='BetaZ2')(BetaZ)
BetaZ = tf.keras.layers.Dense(1, activation = 'linear',name='BetaZ5')(BetaZ)

B = tf.keras.layers.Multiply(name = 'Beta2')([BetaZ,Pain])

y_pred = tf.keras.layers.Add(name = 'model1')([AlphaZ,B])
y_pred = tf.keras.layers.Activation('sigmoid')(y_pred)

Model = tf.keras.models.Model(inputs=[Pain,WOPain], outputs=y_pred)
Model.compile(optimizer='sgd',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

if __name__ == "__main__":
    Model.summary()
