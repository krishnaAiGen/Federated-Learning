#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 21:56:31 2023

@author: krishna
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nRowsRead = None 

df1 = pd.read_csv("Data/cicids-2017/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
df2=pd.read_csv("Data/cicids-2017/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
df3=pd.read_csv("Data/cicids-2017/Friday-WorkingHours-Morning.pcap_ISCX.csv")
df4=pd.read_csv("Data/cicids-2017/Monday-WorkingHours.pcap_ISCX.csv")
df5=pd.read_csv("Data/cicids-2017/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
df6=pd.read_csv("Data/cicids-2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df7=pd.read_csv("Data/cicids-2017/Tuesday-WorkingHours.pcap_ISCX.csv")
df8=pd.read_csv("Data/cicids-2017/Wednesday-workingHours.pcap_ISCX.csv")

df = pd.concat([df1,df2])
del df1,df2
df = pd.concat([df,df3])
del df3
df = pd.concat([df,df4])
del df4
df = pd.concat([df,df5])
del df5
df = pd.concat([df,df6])
del df6
df = pd.concat([df,df7])
del df7
df = pd.concat([df,df8])
del df8

columns = df.columns

df[columns[78]].value_counts()

df = df.sample(frac=1)

print(df.info())
df.isnull().sum()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True) 
print(df.duplicated().sum(), "duplikat baris sepenuhnya untuk dihapus")
df.drop_duplicates(inplace=True)
df.info()

df_encoded = df.copy()
le = LabelEncoder()
df_encoded[' Label'] = le.fit_transform(df[' Label'])



# Split your data into X and y
X = df_encoded.drop(' Label', axis=1)
y = df_encoded[' Label']

X = df.drop(' Label', axis=1)
y = df[' Label']
y = le.fit_transform(y)

#Standard scaling
standard_scaler = StandardScaler()
X = standard_scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

####save_data
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_train.to_csv('cicids_x_train.csv', index = False)
X_test.to_csv('cicids_x_test.csv', index = False)

y_train = np.array(y_train)
y_test = np.array(y_test)

np.save('cicids-y-train', y_train)
np.save('cicids-y-test', y_test)


def load_cicids_data():
    x_train_loaded = pd.read_csv('/Users/krishna/Documents/research_projects/zero-days/Data/cicids-2017/cicids_x_train.csv')
    x_test_loaded = pd.read_csv('/Users/krishna/Documents/research_projects/zero-days/Data/cicids-2017/cicids_x_test.csv')
    y_train_loaded = np.load('/Users/krishna/Documents/research_projects/zero-days/Data/cicids-2017/cicids-y-train.npy')
    y_test_loaded = np.load('/Users/krishna/Documents/research_projects/zero-days/Data/cicids-2017/cicids-y-test.npy')
    
    return x_train_loaded, y_train_loaded, x_test_loaded, y_test_loaded


from tensorflow import keras
model=keras.models.Sequential([
        keras.layers.Flatten(input_shape=[78,]),
        keras.layers.Dense(200,activation='tanh'),
        keras.layers.Dense(100,activation='tanh'),
        keras.layers.Dense(50,activation='tanh'),
        keras.layers.Dense(15,activation='softmax')
    ])

#setting weight of the model
# model.set_weights(self.weights)

#getting the initial weight of the model
# initial_weight=model.get_weights()
# output_weight_list=[]

#training the model
# import animation
# print('###### Client1 Training started ######')
# wait=animation.Wait()
# wait.start()

# import tensorflow as tf
# x_train_loaded, y_train_loaded, x_test_loaded, y_test_loaded = load_cicids_data()
# model.compile(loss='sparse_categorical_crossentropy',optimizer="RMSprop", metrics=['accuracy'])
# history=model.fit(x_test_loaded[:50000], y_test_loaded[:50000],epochs=5,batch_size=64) 
# weights = model.get_weights()

# from collections import Counter
# Counter(y_train)
