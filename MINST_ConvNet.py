#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 19:35:20 2019

@author: changlinli
"""
#%%
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tqdm import tqdm
import tensorflow.contrib.slim as slim

#%% para setup
mb_size=128
p_miss = 0.5
p_hint = 0.9
alpha = 3
Dim=784                     #num of features
Train_No = 5500
Test_No = 1000

#%% data import
mnist = input_data.read_data_sets("MINST_data",one_hot=True)
trainX,_ = mnist.train.next_batch(Train_No)
testX,_ = mnist.test.next_batch(Test_No)

def sample_M(m,n,p):
    A = np.random.uniform(0,1,size=[m,n])
    B = A>p
    C = 1. * B
    return C

##TrainM: Mask of missing for train
##TestM: Mask of missing for test
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

#%% discriminator
#initializer = tf.contrib.layers.xavier_initializer()
#D_W1 = tf.Variable(initializer([None,14,14,32]))  #kernel: 5,  -1*32*32*1>>>-1*14*14*32
#D_b1 = tf.Variable(tf.zeros(shape=[32]))
#
#D_W2 = tf.Variable(initializer([None,7,7,64]))     #kernel: 5,  -1*14*14*32>>-1*7*7*64
#D_b2 = tf.Variable(tf.zeros(shape=[64]))
#
#D_W3 = tf.Variable(initializer([None,4,4,128]))    #Directly create dense layer, check it para
#D_b3 = tf.Variable(tf.zeros(shape=[128]))
#
#theta_D = [D_W1,D_b1,D_W2,D_b2,D_W3,D_b3]

#%% generator
#G_W1 = tf.Variable(initializer([Dim*2,32]))  #kernel: 5,  -1*32*32*1>>>-1*14*14*32
#G_b1 = tf.Variable(tf.zeros(shape=[32]))
#
#G_W2 = tf.Variable(initializer([32,64]))     #kernel: 5,  -1*14*14*32>>-1*7*7*64
#G_b2 = tf.Variable(tf.zeros(shape=[64]))
#
#G_W3 = tf.Variable(initializer([64,128]))    #Directly create dense layer, check it para
#G_b3 = tf.Variable(tf.zeros(shape=[128]))
#
#theta_G = [G_W1,G_b1,G_W2,G_b2,G_W3,G_b3]

#%% def of GAIN

#%% def of generator
def generator(x,z,m):
    ##ADD layer by order: 
    ##Conv, activation, batch normalization and dropout
    # TODO: ADD regulation
    imp = x*m+z*(1-m)
    inputs = tf.concat(values = [imp,m],axis=1) #size:(train_no, 784*2)
    inputs = tf.reshape(inputs, [-1, 28, 28, 2])
    #G_W1 = tf.contrib.layers.fully_connected(inputs,256,activation_fn=tf.nn.sigmoid)
    G_W1 = tf.contrib.layers.conv2d(inputs,num_outputs=32,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.relu)
    G_W1_maxpool = tf.contrib.layers.max_pool2d(G_W1,5)
    G_W2 = tf.contrib.layers.conv2d(G_W1_maxpool,num_outputs=64,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.tanh)
    #G_W3 = tf.contrib.layers.conv2d(G_W2,num_outputs=128,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.tanh)
    G_W3 = tf.contrib.layers.flatten(G_W2)
    G_prob = tf.contrib.layers.fully_connected(G_W3,1024,activation_fn=tf.nn.sigmoid)
    G_prob2 = tf.contrib.layers.fully_connected(G_prob,Dim,activation_fn=tf.nn.sigmoid)
    ##reshape G_W2,countinue line G_prob
    return G_prob2

def discriminator(x,m,g,h):
    imp = m*x + (1-m)*g
    inputs = tf.concat(axis=1,values=[imp,h])
    inputs = tf.reshape(inputs, [-1, 28, 28, 2])
    #D_W1 = tf.contrib.layers.fully_connected(inputs,256,activation_fn=tf.nn.sigmoid)
    D_W1 = tf.contrib.layers.conv2d(inputs,num_outputs=32,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.relu)
    D_W1_maxpool = tf.contrib.layers.max_pool2d(D_W1,5)
    D_W2 = tf.contrib.layers.conv2d(D_W1_maxpool,num_outputs=64,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.tanh)
    #G_W3 = tf.contrib.layers.conv2d(G_W2,num_outputs=128,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.tanh)
    D_W3 = tf.contrib.layers.flatten(D_W2)
    D_prob = tf.contrib.layers.fully_connected(D_W3,1024,activation_fn=tf.nn.sigmoid)
    D_prob2 = tf.contrib.layers.fully_connected(D_prob,Dim,activation_fn=tf.nn.sigmoid)
    return D_prob2

#def generator2(x,z,m):
#    ##ADD layer by order: 
#    ##Conv, activation, batch normalization and dropout
#    # TODO: ADD regulation
#    imp = x*m+z*(1-m)
#    inputs = tf.concat(values = [imp,m],axis=1) #size:(train_no, 784*2)
#    #inputs = tf.reshape(inputs, [-1, 28, 28, 2])
#    G_W1 = tf.contrib.layers.fully_connected(inputs,256,activation_fn=tf.nn.sigmoid)
#    #G_W1 = tf.contrib.layers.conv2d(inputs,num_outputs=32,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.relu)
#    #G_W1_maxpool = tf.contrib.layers.max_pool2d(G_W1,7)
#    #G_W2 = tf.contrib.layers.conv2d(G_W1,num_outputs=64,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.tanh)
#    #G_W3 = tf.contrib.layers.conv2d(G_W2,num_outputs=128,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.tanh)
#    #G_W3 = tf.contrib.layers.flatten(G_W1)
#    G_prob = tf.contrib.layers.fully_connected(G_W1,128,activation_fn=tf.nn.sigmoid)
#    G_prob2 = tf.contrib.layers.fully_connected(G_prob,Dim,activation_fn=tf.nn.sigmoid)
#    ##reshape G_W2,countinue line G_prob
#    return G_prob2
#
#def discriminator2(x,m,g,h):
#    imp = m*x + (1-m)*g
#    inputs = tf.concat(axis=1,values=[imp,h])
#    #inputs = tf.reshape(inputs, [-1, 28, 28, 2])
#    D_W1 = tf.contrib.layers.fully_connected(inputs,256,activation_fn=tf.nn.sigmoid)
#    #D_W1 = tf.contrib.layers.conv2d(inputs,num_outputs=32,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.relu)
#    #D_W1_maxpool = tf.contrib.layers.max_pool2d(D_W1,7)
#    #D_W2 = tf.contrib.layers.conv2d(D_W1,num_outputs=64,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.tanh)
#    #G_W3 = tf.contrib.layers.conv2d(G_W2,num_outputs=128,kernel_size=5,normalizer_fn = tf.contrib.layers.batch_norm, activation_fn=tf.nn.tanh)
#    #D_W3 = tf.contrib.layers.flatten(D_W1)
#    D_prob = tf.contrib.layers.fully_connected(D_W1,128,activation_fn=tf.nn.sigmoid)
#    D_prob2 = tf.contrib.layers.fully_connected(D_prob,Dim,activation_fn=tf.nn.sigmoid)
#    return D_prob2

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
#%% setup solver
#D_solver = tf.train.AdamOptimizer().minimize(D_loss,var_list = theta_D)
#G_solver = tf.train.AdamOptimizer().minimize(G_loss,var_list = theta_G)
D_solver = tf.train.AdamOptimizer().minimize(D_loss)
G_solver = tf.train.AdamOptimizer().minimize(G_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#%% output initialization
if not os.path.exists('Multiple_Impute_out1/'):
    os.makedirs('Multiple_Impute_out1/')

i=1
turns = 10001
for it in tqdm(range(turns)):
    ## Inputs
    mb_idx = sample_idx(Train_No,mb_size)
    X_mb = trainX[mb_idx,:]
    Z_mb = sample_Z(mb_size,Dim)
    M_mb = trainM[mb_idx,:]
    H_mb1 = sample_M(mb_size,Dim,1-p_hint)
    H_mb = M_mb*H_mb1
    
    New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
    _, D_loss_curr = sess.run([D_solver, D_loss1], feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb, H: H_mb})
    _, G_loss_curr, MSE_train_loss_curr, MSE_test_loss_curr = sess.run([G_solver, G_loss1, MSE_train_loss, MSE_test_loss],
                                                                       feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb, H: H_mb})

    #%% Output figure
    if it % 100 == 0:
      
        mb_idx = sample_idx(Test_No, 5)
        X_mb = testX[mb_idx,:]
        M_mb = testM[mb_idx,:]  
        Z_mb = sample_Z(5, Dim) 
    
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
        
        samples1 = X_mb                
        samples5 = M_mb * X_mb + (1-M_mb) * Z_mb
        
        samples2 = sess.run(G_sample, feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb})
        samples2 = M_mb * X_mb + (1-M_mb) * samples2        
        
        Z_mb = sample_Z(5, Dim) 
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb       
        samples3 = sess.run(G_sample, feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb})
        samples3 = M_mb * X_mb + (1-M_mb) * samples3     
        
        Z_mb = sample_Z(5, Dim) 
        New_X_mb = M_mb * X_mb + (1-M_mb) * Z_mb       
        samples4 = sess.run(G_sample, feed_dict = {X: X_mb, M: M_mb, Z: New_X_mb})
        samples4 = M_mb * X_mb + (1-M_mb) * samples4     
        
        samples = np.vstack([samples5, samples2, samples3, samples4, samples1])          
        
        fig = plot(samples)
        plt.savefig('Multiple_Impute_out1/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
        
    #%% Intermediate Losses
    if it % 100 == 0:
        print('Iter: {}'.format(it))
        print('Train_loss: {:.4}'.format(MSE_train_loss_curr))
        print('Test_loss: {:.4}'.format(MSE_test_loss_curr))
        print('D_loss: {:.4}'.format(D_loss_curr))