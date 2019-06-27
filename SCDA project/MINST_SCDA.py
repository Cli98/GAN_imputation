#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:35:20 2019

@author: changlinli
"""
#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tqdm import tqdm
import random
from time import time
import gzip

from matplotlib import pyplot as plt
import pylab as pl
import numpy as np
import pandas as pd
import matplotlib.cm as cm

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#%% para setup
mb_size=32
p_miss = 0.05
p_hint = 0.9
alpha = 5
seed = 28213
#%% data import
input_name = 'genotype_full_1_2.txt'  # 'data/genotype_full.txt'
df_ori = pd.read_csv(input_name, sep='\t', index_col=0)
print('data shape:', df_ori.shape)
# output dir
output_path = 'models'
#%% def mask
def sample_M(m,n,p):
    A = np.random.uniform(0,1,size=[m,n])
    B = A>p
    C = 1. * B
    return C

##TrainM: Mask of missing for train
##TestM: Mask of missing for test

df = df_ori.iloc[0:1000:, 0:5000]
Train_No = int(len(df)*0.65)
validation_No = int(len(df)*0.15)
Test_No = int(len(df)*0.20)
trainX = df.iloc[:Train_No,:]
Dim= df.shape[1]                   #num of features
# hyperparameters
missing_perc = 0.05
feature_size = df.shape[1]
lr = 0.1
epochs = 10
trainM = sample_M(Train_No,Dim,p_miss)
testM = sample_M(Test_No,Dim,p_miss)

#%% function def
def plot(samples):
    fig = plt.figure(figsize = (5,5))
    gs = gridspec.GridSpec(5,5)
    gs.update(wspace=0.05,hspace=0.05)
    
    for i,sample in enumerate(samples):
        ax=plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28,28),cmap = 'Greys_r')
    return fig

#%% placeholder
X = tf.placeholder(tf.float32,shape = [None,Dim])   #X for input
M = tf.placeholder(tf.float32,shape = [None,Dim])   #M for mask
H = tf.placeholder(tf.float32,shape = [None,Dim])   #H for hint
Z = tf.placeholder(tf.float32,shape = [None,Dim])   #Z for random noise
y_true = tf.placeholder(tf.float32, shape=[None, Dim], name='y_true')
#%% discriminator
#initializer = tf.contrib.layers.xavier_initializer()
#
#D_W1 = tf.Variable(initializer([Dim*2,2048]))
#D_b1 = tf.Variable(tf.zeros(shape=[2048]))
#
#D_W2 = tf.Variable(initializer([2048,1024]))
#D_b2 = tf.Variable(tf.zeros(shape=[1024]))
#
#D_W3 = tf.Variable(initializer([1024,Dim]))
#D_b3 = tf.Variable(tf.zeros(shape=[Dim]))
#
#theta_D = [D_W1,D_b1,D_W2,D_b2,D_W3,D_b3]

#%% generator

#G_W1 = tf.Variable(initializer([Dim*2,2048]))
#G_b1 = tf.Variable(tf.zeros(shape=[2048]))
#
#G_W2 = tf.Variable(initializer([2048,1024]))
#G_b2 = tf.Variable(tf.zeros(shape=[1024]))
#
#G_W3 = tf.Variable(initializer([1024,Dim]))
#G_b3 = tf.Variable(tf.zeros(shape=[Dim]))
#
#theta_G = [G_W1,G_b1,G_W2,G_b2,G_W3,G_b3]
#%% recommendation para
##epoch>> 10 or 20
##minibatch size= 32, try all those then 1, 2, 4, 8, 16, 32, 64, 128, 256
##learning rate>>>0.01
#%% def of GAIN hyperpara
NUM_OUT1 = 2048*5
NUM_OUT2 = 1024*7
kr = 0.0001
keep = 0.50
#%% def of generator
def generator(x,z,m):
    imp = x*m+z*(1-m)
    inputs = tf.concat(values = [imp,m],axis=1)
    G_W1 = tf.contrib.layers.fully_connected(inputs,NUM_OUT1,weights_regularizer=tf.contrib.layers.l1_regularizer(kr))
    G_W1_loss = tf.contrib.layers.dropout(G_W1,keep_prob=keep)
    G_W2 = tf.contrib.layers.fully_connected(G_W1_loss,NUM_OUT2,weights_regularizer=tf.contrib.layers.l1_regularizer(kr))
    G_W2_loss = tf.contrib.layers.dropout(G_W2,keep_prob=keep)
    G_prob = tf.contrib.layers.fully_connected(G_W2_loss,Dim,activation_fn=tf.nn.sigmoid,weights_regularizer=tf.contrib.layers.l1_regularizer(kr))
    return G_prob

def discriminator(x,m,g,h):
    imp = m*x + (1-m)*g
    inputs = tf.concat(axis=1,values=[imp,h])
    D_W1 = tf.contrib.layers.fully_connected(inputs,NUM_OUT1,weights_regularizer=tf.contrib.layers.l1_regularizer(kr))
    D_W1_loss = tf.contrib.layers.dropout(D_W1,keep_prob=keep)
    D_W2 = tf.contrib.layers.fully_connected(D_W1_loss,NUM_OUT2,weights_regularizer=tf.contrib.layers.l1_regularizer(kr))
    D_W2_loss = tf.contrib.layers.dropout(D_W2,keep_prob=keep)
    D_prob = tf.contrib.layers.fully_connected(D_W2_loss,Dim,activation_fn=tf.nn.sigmoid,weights_regularizer=tf.contrib.layers.l2_regularizer(kr))
    return D_prob

def sample_Z(m,n):
    return np.random.uniform(0.,1.,size=[m,n])

def sample_idx(m,n):
    A = np.random.permutation(m)
    idx = A[:n]
    return idx

#%% Structure
G_sample = generator(X,Z,M)
D_prob = discriminator(X,M,G_sample,H)

#%% Loss
##D_loss may need to change
D_loss1 = -tf.reduce_mean(M*tf.log(D_prob + 1e-8)+(1-M)*tf.log(1-D_prob+1e-8))
G_loss1 = -tf.reduce_mean((1-M)*tf.log(D_prob+1e-8))/tf.reduce_mean(1-M)
MSE_train_loss = tf.reduce_mean((M*X-M*G_sample)**2)/tf.reduce_mean(M)

D_loss = D_loss1
G_loss = G_loss1 + alpha*MSE_train_loss

MSE_test_loss = tf.reduce_mean(((1-M)*X-(1-M)*G_sample)**2)/tf.reduce_mean(1-M)

y_pred = tf.nn.softmax(G_sample, name='output')
logits = tf.argmax(y_pred, axis=1)
CV_loss, CV_loss_op = tf.metrics.accuracy(labels=tf.argmax(y_true, axis=1), predictions=tf.argmax(y_pred, 1))

#%% setup solver
D_solver = tf.train.AdamOptimizer().minimize(D_loss)
G_solver = tf.train.AdamOptimizer().minimize(G_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
#%% output initialization
if not os.path.exists('Multiple_Impute_out1/'):
    os.makedirs('Multiple_Impute_out1/')
i=1

for it in tqdm(range(10000)):
    ## Inputs
    mb_idx = sample_idx(Train_No,mb_size)
    X_mb = trainX.iloc[mb_idx,:]
    Z_mb = sample_Z(mb_size,Dim)
    M_mb = trainM[mb_idx,:]
    H_mb1 = sample_M(mb_size,Dim,1-p_hint)
    H_mb = M_mb*H_mb1
    
    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
    _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb, H: H_mb})
    _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr,Cv_loss = sess.run([G_solver, G_loss1, MSE_train_loss, MSE_test_loss,CV_loss_op],
                                                                       feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb, H: H_mb, y_true :X_mb})
            
    #%% Output figure
#    if it % 100 == 0:
#      
#        mb_idx = sample_idx(Test_No, 5)
#        X_mb = testX[mb_idx,:]
#        M_mb = testM[mb_idx,:]  
#        Z_mb = sample_Z(5, Dim) 
#    
#        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
#        
#        samples1 = X_mb                
#        samples5 = M_mb * X_mb + (1-M_mb) * Z_mb
#        
#        samples2 = sess.run(G_sample, feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb})
#        samples2 = M_mb * X_mb + (1-M_mb) * samples2        
#        
#        Z_mb = sample_Z(5, Dim) 
#        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb       
#        samples3 = sess.run(G_sample, feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb})
#        samples3 = M_mb * X_mb + (1-M_mb) * samples3     
#        
#        Z_mb = sample_Z(5, Dim) 
#        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb       
#        samples4 = sess.run(G_sample, feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb})
#        samples4 = M_mb * X_mb + (1-M_mb) * samples4     
#        
#        samples = np.vstack([samples5, samples2, samples3, samples4, samples1])          
#        
#        fig = plot(samples)
#        plt.savefig('Multiple_Impute_out1/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
#        i += 1
#        plt.close(fig)
        
    #%% Intermediate Losses
    if it % 100 == 0:
        print('\nIter: {}'.format(it))
        print('Train_loss: {:.4}'.format(MSE_train_loss_curr))
        print('Test_loss: {:.4}'.format(MSE_test_loss_curr))
        print('CV_loss: {:.4}'.format(Cv_loss))
        print('D_loss: {:.4}'.format(D_loss_curr))