# -*- coding: utf-8 -*-


#######################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras import Sequential

import kagglehub
import os
path = kagglehub.dataset_download("shebrahimi/financial-distress")


print("Path to dataset files:", path)#여기서의 반환값을 아래 train_dir에 투입

train_dir = "/root/.cache/kagglehub/datasets/shebrahimi/financial-distress/versions/1"
#train_dir = "/kaggle/input/financial-distress"
dt = pd.read_csv(os.path.join(train_dir, os.listdir(train_dir)[0]))

dt['Distress'] = np.array([0 if dt['Financial Distress'][i] > -0.5 else 1 for i in range(dt.shape[0])])
dt.drop(['Financial Distress'], axis=1, inplace=True)
dt.drop(['Company','Time'], axis=1, inplace=True)


X = dt.iloc[:,0:83]
y = dt.iloc[:,-1]


X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.3,shuffle=True, random_state=321)
print(X_tr.shape, np.bincount(y_tr), np.bincount(y_te))


#Custom Pruning class definition

class Pruning_one(tf.keras.Model):
  def __init__(self,alpha,W_t,d):
    super(Pruning_one, self).__init__()
    self.alpha = alpha
    self.W_t = W_t
    self.d = d

    mat = []; thres_1 =[]
    for i in range(self.d):
      w = self.W_t[i]; w = np.array(w);w_ = np.abs(w.reshape(-1))
      w_1 = w_[w_ > 0]
      thres = np.quantile(np.abs(w_1),self.alpha)
      mk = np.where(np.abs(w)>thres,1,0)
      mat.append(mk)
      thres_1.append(thres)

    self.mat = mat
    self.thres = thres_1

  def params(self):
    return self.mat, self.thres


  def pruning_const(self,k):
    return self.mat[k]


#Baseline Model Fitting

tf.keras.utils.set_random_seed(678)
md1 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(83,)),
    tf.keras.layers.Dense(20,activation="relu",kernel_initializer="glorot_uniform",use_bias=False),
    tf.keras.layers.Dense(10,activation="relu",kernel_initializer="glorot_uniform",use_bias=False),
    tf.keras.layers.Dense(10,activation="relu",kernel_initializer="glorot_uniform",use_bias=False),
    tf.keras.layers.Dense(1,activation="sigmoid",use_bias=False)
])

md1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True),metrics=['accuracy'])

hist = md1.fit(X_tr,y_tr,epochs=200,batch_size=200)

_, result1 = md1.evaluate(X_te,y_te,verbose=0)
print(result1)

prune = Pruning_one(0.2,md1.weights, 4)

prune.pruning_const(0)

A,B = prune.params()


###################Pruning Step

Epochs=3
W_ = md1.weights

for j in range(Epochs):
  prune = Pruning_one(0.2,W_, 4)

  class mask_1(tf.keras.constraints.Constraint):
    def __call__(self,w):
      return tf.convert_to_tensor(prune.pruning_const(0))*w

  class mask_2(tf.keras.constraints.Constraint):
    def __call__(self,w):
      return tf.convert_to_tensor(prune.pruning_const(1))*w

  class mask_3(tf.keras.constraints.Constraint):
    def __call__(self,w):
      return tf.convert_to_tensor(prune.pruning_const(2))*w

  class mask_4(tf.keras.constraints.Constraint):
    def __call__(self,w):
      return tf.convert_to_tensor(prune.pruning_const(3))*w

  md2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(shape=(83,)),
    tf.keras.layers.Dense(20,activation="relu",use_bias=False,kernel_constraint=mask_1()),
    tf.keras.layers.Dense(10,activation="relu",use_bias=False,kernel_constraint=mask_2()),
    tf.keras.layers.Dense(10,activation="relu",use_bias=False,kernel_constraint=mask_3()),
    tf.keras.layers.Dense(1,activation="sigmoid",use_bias=False,kernel_constraint=mask_4())
    ])

  starting1 = tf.constant(W_[0], dtype=tf.float32)
  starting2 = tf.constant(W_[1], dtype=tf.float32)
  starting3 = tf.constant(W_[2], dtype=tf.float32)
  starting4 = tf.constant(W_[3], dtype=tf.float32)

  md2.layers[0].set_weights([starting1])
  md2.layers[1].set_weights([starting2])
  md2.layers[2].set_weights([starting3])
  md2.layers[3].set_weights([starting4])

  md2.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True),metrics=['accuracy'])
  hist2 = md2.fit(X_tr,y_tr,epochs=20,batch_size=200)
  W_ = md2.weights

_2, result2 = md2.evaluate(X_te,y_te,verbose=0)
print(result2)


print(len(np.abs(np.array(md1.weights[0]).reshape(-1))[np.abs(np.array(md1.weights[0]).reshape(-1))>0]))
print(len(np.abs(np.array(md1.weights[1]).reshape(-1))[np.abs(np.array(md1.weights[1]).reshape(-1))>0]))
print(len(np.abs(np.array(md1.weights[2]).reshape(-1))[np.abs(np.array(md1.weights[2]).reshape(-1))>0]))
print(len(np.abs(np.array(md1.weights[3]).reshape(-1))[np.abs(np.array(md1.weights[3]).reshape(-1))>0]))

print(len(np.abs(np.array(md2.weights[0]).reshape(-1))[np.abs(np.array(md2.weights[0]).reshape(-1))>0]))
print(len(np.abs(np.array(md2.weights[1]).reshape(-1))[np.abs(np.array(md2.weights[1]).reshape(-1))>0]))
print(len(np.abs(np.array(md2.weights[2]).reshape(-1))[np.abs(np.array(md2.weights[2]).reshape(-1))>0]))
print(len(np.abs(np.array(md2.weights[3]).reshape(-1))[np.abs(np.array(md2.weights[3]).reshape(-1))>0]))

