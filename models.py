#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 02:33:42 2023

@author: krishna
"""


from tensorflow import keras

def get_cicids_model():
    model=keras.models.Sequential([
    keras.Input(shape=[78,]),
    keras.layers.Flatten(),
    keras.layers.Dense(200,activation='tanh'),
    keras.layers.Dense(100,activation='tanh'),
    keras.layers.Dense(50,activation='tanh'),
    keras.layers.Dense(15,activation='softmax')
    ])
    
    
    return model

# def get_cicids_model():
#     # Define a constant initializer with the desired weight value, e.g., 0.1
#     constant_initializer = tf.keras.initializers.Constant(value=0.1)
    
#     model = keras.models.Sequential([
#         keras.Input(shape=[78,]),
#         keras.layers.Flatten(),
#         keras.layers.Dense(200, activation='tanh', 
#                            kernel_initializer=constant_initializer, 
#                            bias_initializer=constant_initializer),
#         keras.layers.Dense(100, activation='tanh', 
#                            kernel_initializer=constant_initializer, 
#                            bias_initializer=constant_initializer),
#         keras.layers.Dense(50, activation='tanh', 
#                            kernel_initializer=constant_initializer, 
#                            bias_initializer=constant_initializer),
#         keras.layers.Dense(15, activation='softmax', 
#                            kernel_initializer=constant_initializer, 
#                            bias_initializer=constant_initializer)
#     ])
    
#     return model

def get_nslkdd_model():
    model=keras.models.Sequential([
    keras.Input(shape=[122,]),
    keras.layers.Flatten(),
    keras.layers.Dense(200,activation='tanh'),
    keras.layers.Dense(100,activation='tanh'),
    keras.layers.Dense(5,activation='softmax')
    ])
    
    return model
    

