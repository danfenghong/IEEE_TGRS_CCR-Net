# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:46:19 2018

@author: danfeng
"""
import numpy as np
#import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as scio 
import scipy.io as sio
from tf_utils import random_mini_batches_standardtwoModality, convert_to_one_hot
from tensorflow.python.framework import ops
#from tfdeterminism import patch
#patch() %If you would like to get a fixed result in each running the network, you can open it.

def create_placeholders(n_x1, n_x2, n_y):
   
    isTraining = tf.placeholder_with_default(True, shape=())
    x1 = tf.placeholder(tf.float32, [None, n_x1], name = "x1")
    x2 = tf.placeholder(tf.float32, [None, n_x2], name = "x2")
    y = tf.placeholder(tf.float32, [None, n_y], name = "Y")
 
    return x1, x2, y, isTraining


def initialize_parameters():

    
    tf.set_random_seed(1)
     
    x1_conv_w1 = tf.get_variable("x1_conv_w1", [3,3,144,16], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    #x1_conv_w11 = tf.get_variable("x1_conv_w11", [1,1,144*16,16], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_conv_b1 = tf.get_variable("x1_conv_b1", [16], initializer = tf.zeros_initializer())
    x2_conv_w1 = tf.get_variable("x2_conv_w1", [3,3,1,16], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_conv_b1 = tf.get_variable("x2_conv_b1", [16], initializer = tf.zeros_initializer())
    
    x1_conv_w2 = tf.get_variable("x1_conv_w2", [3,3,16,32], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x1_conv_b2 = tf.get_variable("x1_conv_b2", [32], initializer = tf.zeros_initializer())
    x2_conv_w2 = tf.get_variable("x2_conv_w2", [3,3,16,32], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x2_conv_b2 = tf.get_variable("x2_conv_b2", [32], initializer = tf.zeros_initializer())
    
    x_en_w1 = tf.get_variable("x_en_w1", [32*8,64], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x_en_b1 = tf.get_variable("x_en_b1", [64], initializer = tf.zeros_initializer())
    
    x_en_w2 = tf.get_variable("x_en_w2", [64,32], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x_en_b2 = tf.get_variable("x_en_b2", [32], initializer = tf.zeros_initializer())
    
    x_en_w3 = tf.get_variable("x_en_w3", [32,32], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x_en_b3 = tf.get_variable("x_en_b3", [32], initializer = tf.zeros_initializer())

    x_de_w1 = tf.get_variable("x_de_w1", [32,32], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x_de_b1 = tf.get_variable("x_de_b1", [32], initializer = tf.zeros_initializer())
    
    x_de_w2 = tf.get_variable("x_de_w2", [32,64], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x_de_b2 = tf.get_variable("x_de_b2", [64], initializer = tf.zeros_initializer())
    
    x_de_w3 = tf.get_variable("x_de_w3", [64,32*8], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x_de_b3 = tf.get_variable("x_de_b3", [32*8], initializer = tf.zeros_initializer())   

    x_en_w4 = tf.get_variable("x_en_w4", [32,15], initializer = tf.contrib.layers.variance_scaling_initializer(seed = 1))
    x_en_b4 = tf.get_variable("x_en_b4", [15], initializer = tf.zeros_initializer())   
    
    parameters = {"x1_conv_w1": x1_conv_w1,
                  #"x1_conv_w11": x1_conv_w11,
                  "x1_conv_b1": x1_conv_b1,
                  "x2_conv_w1": x2_conv_w1,
                  "x2_conv_b1": x2_conv_b1,
                  "x1_conv_w2": x1_conv_w2,
                  "x1_conv_b2": x1_conv_b2,
                  "x2_conv_w2": x2_conv_w2,
                  "x2_conv_b2": x2_conv_b2,
                  "x_en_w1": x_en_w1,
                  "x_en_b1": x_en_b1,
                  "x_en_w2": x_en_w2,
                  "x_en_b2": x_en_b2,
                  "x_en_w3": x_en_w3,
                  "x_en_b3": x_en_b3,
                  "x_de_w1": x_de_w1,
                  "x_de_b1": x_de_b1,
                  "x_de_w2": x_de_w2,
                  "x_de_b2": x_de_b2,
                  "x_de_w3": x_de_w3,
                  "x_de_b3": x_de_b3,
                  "x_en_w4": x_en_w4,
                  "x_en_b4": x_en_b4}   
    
    return parameters


def mynetwork(x1, x2, parameters, isTraining):

    
    x1 = tf.reshape(x1, [-1, 7, 7, 144], name = "x1")
    x2 = tf.reshape(x2, [-1, 7, 7, 1], name = "x2")
    
    with tf.name_scope("encoder_layer_1"):
         
         x1_conv_layer_z1 = tf.nn.conv2d(x1, parameters['x1_conv_w1'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b1']                                  
         #x1_conv_layer_z1 = tf.nn.separable_conv2d(x1, parameters['x1_conv_w1'], parameters['x1_conv_w11'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b1']
         x1_conv_layer_z1_bn = tf.layers.batch_normalization(x1_conv_layer_z1, momentum = 0.9, training = isTraining)  
         x1_conv_layer_z1_po = tf.layers.max_pooling2d(x1_conv_layer_z1_bn, 2, 2, padding='SAME')
         x1_conv_layer_a1 = tf.nn.relu(x1_conv_layer_z1_po)
          
         x2_conv_layer_z1 = tf.nn.conv2d(x2, parameters['x2_conv_w1'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x2_conv_b1']                                  
         x2_conv_layer_z1_bn = tf.layers.batch_normalization(x2_conv_layer_z1, momentum = 0.9, training = isTraining)  
         x2_conv_layer_z1_po = tf.layers.max_pooling2d(x2_conv_layer_z1_bn, 2, 2, padding='SAME')
         x2_conv_layer_a1 = tf.nn.relu(x2_conv_layer_z1_po)

    with tf.name_scope("encoder_layer_2"):
         
         x1_conv_layer_z2 = tf.nn.conv2d(x1_conv_layer_a1, parameters['x1_conv_w2'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x1_conv_b2']   
         x1_conv_layer_z2_bn = tf.layers.batch_normalization(x1_conv_layer_z2, momentum = 0.9, training = isTraining)                                             
         x1_conv_layer_z2_po = tf.layers.max_pooling2d(x1_conv_layer_z2_bn, 2, 2, padding='SAME')
         x1_conv_layer_a2 = tf.nn.relu(x1_conv_layer_z2_po)
 
         x2_conv_layer_z2 = tf.nn.conv2d(x2_conv_layer_a1, parameters['x2_conv_w2'], strides=[1, 1, 1, 1], padding='SAME') + parameters['x2_conv_b2']   
         x2_conv_layer_z2_bn = tf.layers.batch_normalization(x2_conv_layer_z2, momentum = 0.9, training = isTraining)                                             
         x2_conv_layer_z2_po = tf.layers.max_pooling2d(x2_conv_layer_z2_bn, 2, 2, padding='SAME')
         x2_conv_layer_a2 = tf.nn.relu(x2_conv_layer_z2_po)
         
         x1_conv_layer_a2_shape = x1_conv_layer_a2.get_shape().as_list()
         x1_conv_layer_a2_2d = tf.reshape(x1_conv_layer_a2, [-1, x1_conv_layer_a2_shape[1] * x1_conv_layer_a2_shape[2] * x1_conv_layer_a2_shape[3]])        
         x2_conv_layer_a2_shape = x2_conv_layer_a2.get_shape().as_list()
         x2_conv_layer_a2_2d = tf.reshape(x2_conv_layer_a2, [-1, x2_conv_layer_a2_shape[1] * x2_conv_layer_a2_shape[2] * x2_conv_layer_a2_shape[3]])
         
         joint_layer = tf.concat([x1_conv_layer_a2_2d, x2_conv_layer_a2_2d], 1)
         joint_layer_re = tf.concat([x2_conv_layer_a2_2d, x1_conv_layer_a2_2d], 1)
         
    with tf.name_scope("encoder_layer_3"):     
         
         x1_en_z1 = tf.matmul(joint_layer, parameters['x_en_w1']) + parameters['x_en_b1']   
         x1_en_z1_bn = tf.layers.batch_normalization(x1_en_z1, momentum = 0.9, training = isTraining)                                           
         x1_en_a1 = tf.nn.relu(x1_en_z1_bn)

    with tf.name_scope("decoder_layer_3"):
         
         x1_de_z3 = tf.matmul(x1_en_a1, parameters['x_de_w3']) + parameters['x_de_b3']     
         x1_de_z3_bn = tf.layers.batch_normalization(x1_de_z3, momentum = 0.9, training = isTraining)      
         x1_de_a3 = tf.nn.relu(x1_de_z3_bn)  
            
    with tf.name_scope("encoder_layer_4"):
                
         x1_en_z2 = tf.matmul(x1_en_a1, parameters['x_en_w2']) + parameters['x_en_b2']     
         x1_en_z2_bn = tf.layers.batch_normalization(x1_en_z2, momentum = 0.9, training = isTraining)      
         x1_en_a2 = tf.nn.relu(x1_en_z2_bn)   

    with tf.name_scope("decoder_layer_2"):
         
         x1_de_z2 = tf.matmul(x1_en_a2, parameters['x_de_w2']) + parameters['x_de_b2']     
         x1_de_z2_bn = tf.layers.batch_normalization(x1_de_z2, momentum = 0.9, training = isTraining)      
         x1_de_a2 = tf.nn.relu(x1_de_z2_bn)  

         joint_layer_re1 = tf.concat([x1_en_a1[:, 32: 64], x1_en_a1[:, 0 : 32]], 1)
         
    with tf.name_scope("encoder_layer_5"):
        
         x1_en_z3 = tf.matmul(x1_en_a2, parameters['x_en_w3']) + parameters['x_en_b3']     
         x1_en_z3_bn = tf.layers.batch_normalization(x1_en_z3, momentum = 0.9, training = isTraining)      
         x1_en_a3 = tf.nn.relu(x1_en_z3_bn)  
         
    with tf.name_scope("decoder_layer_1"):
                
         x1_de_z1 = tf.matmul(x1_en_a3, parameters['x_de_w1']) + parameters['x_de_b1']     
         x1_de_z1_bn = tf.layers.batch_normalization(x1_de_z1, momentum = 0.9, training = isTraining)      
         x1_de_a1 = tf.nn.relu(x1_de_z1_bn)   

         joint_layer_re2 = tf.concat([x1_en_a2[:, 16: 32], x1_en_a1[:, 0 : 16]], 1)
         
    with tf.name_scope("encoder_layer_6"):
        
         x_en_z4 = tf.matmul(x1_en_a3, parameters['x_en_w4']) + parameters['x_en_b4']

    l2_loss =  tf.nn.l2_loss(parameters['x1_conv_w1']) + tf.nn.l2_loss(parameters['x1_conv_w2']) + tf.nn.l2_loss(parameters['x2_conv_w1']) + tf.nn.l2_loss(parameters['x2_conv_w2'])\
               + tf.nn.l2_loss(parameters['x_en_w1']) + tf.nn.l2_loss(parameters['x_en_w2']) + tf.nn.l2_loss(parameters['x_en_w3']) + tf.nn.l2_loss(parameters['x_en_w4'])\
               + tf.nn.l2_loss(parameters['x_de_w1']) + tf.nn.l2_loss(parameters['x_de_w2']) + tf.nn.l2_loss(parameters['x_de_w3'])
    
    return x_en_z4, joint_layer_re, x1_de_a3, joint_layer_re1, x1_de_a2, joint_layer_re2, x1_de_a1, l2_loss

def mynetwork_optimaization(y_es, y_re, r1, r2, r11, r22, r111, r222, l2_loss, reg, learning_rate, global_step):

    with tf.name_scope("cost"):
        
         cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y_es, labels = y_re)) + reg * l2_loss\
                + 0.1 * tf.reduce_mean(tf.square(r1 - r2)) + 0.1 * tf.reduce_mean(tf.square(r11 - r22)) + 0.1 * tf.reduce_mean(tf.square(r111 - r222))

    with tf.name_scope("optimization"):
         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost, global_step=global_step)
         optimizer = tf.group([optimizer, update_ops])
         
    return cost, optimizer

def train_mynetwork(x1_train_set, x2_train_set, x1_test_set, x2_test_set, y_train_set, y_test_set,
           learning_rate_base = 0.001, beta_reg = 0.1, num_epochs = 200, minibatch_size = 32, print_cost = True):
    
    ops.reset_default_graph()                       
    tf.set_random_seed(1)                          
    seed = 1                                     
    (m, n_x1) = x1_train_set.shape                        
    (m, n_x2) = x2_train_set.shape
    (m, n_y) = y_train_set.shape                                    

    costs = []                                   
    costs_dev = []
    train_acc = []
    val_acc = []
    correct_prediction = 0
    
    # Create Placeholders of shape (n_x, n_y)
    x1, x2, y, isTraining = create_placeholders(n_x1, n_x2, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    with tf.name_scope("network"):

         joint_layer, r1, r2, r11, r22, r111, r222, l2_loss = mynetwork(x1, x2, parameters, isTraining)
         
    global_step = tf.Variable(0, trainable = False)
    learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, 30 * m/minibatch_size, 0.5, staircase = True)
    #learning_rate = learning_rate_base
    with tf.name_scope("optimization"):
         # network optimization
         cost, optimizer = mynetwork_optimaization(joint_layer, y, r1, r2, r11, r22, r111, r222, l2_loss, beta_reg, learning_rate, global_step)

    with tf.name_scope("metrics"):
         # Calculate the correct predictions
         joint_layerT = tf.transpose(joint_layer)
         yT = tf.transpose(y)
         correct_prediction = tf.equal(tf.argmax(joint_layerT), tf.argmax(yT))
         accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Initialize all the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs + 1):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            epoch_acc = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches_standardtwoModality(x1_train_set, x2_train_set, y_train_set, minibatch_size, seed)
            for minibatch in minibatches:

                # Select a minibatch
                (batch_x1, batch_x2, batch_y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _, minibatch_cost, minibatch_acc = sess.run([optimizer, cost, accuracy], feed_dict={x1: batch_x1, x2: batch_x2, y: batch_y, isTraining: True})
           
                epoch_cost += minibatch_cost / (num_minibatches+ 1)
                epoch_acc += minibatch_acc / (num_minibatches + 1)
            '''
            minibatches_TE = random_mini_batches_GCN1(x1_train_set, x2_train_set, y_train_set, mask_TR, minibatch_size, seed)
            for minibatch in minibatches:
            '''    
            feature, epoch_cost_dev, epoch_acc_dev = sess.run([joint_layerT, cost, accuracy], feed_dict={x1: x1_test_set, x2: x2_test_set, y: y_test_set, isTraining: False})

            # Print the cost every epoch
            if print_cost == True and epoch % 50 == 0:
                print ("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (epoch, epoch_cost, epoch_cost_dev, epoch_acc, epoch_acc_dev))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                train_acc.append(epoch_acc)
                costs_dev.append(epoch_cost_dev)
                val_acc.append(epoch_acc_dev)
        
        plt.plot(np.squeeze(costs))
        plt.plot(np.squeeze(costs_dev))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # plot the accuracy 
        plt.plot(np.squeeze(train_acc))
        plt.plot(np.squeeze(val_acc))
        plt.ylabel('accuracy')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        
        print("save model")
        save_path = saver.save(sess,"D:\Python_Project\IGARSS2021/save_MuCNN/model.ckpt")
        print("save model:{0} Finished".format(save_path))
       
        return parameters, val_acc, feature


HSI = scio.loadmat('Houston2013_Data/Houston2013_HSI.mat')
HSI = HSI['HSI']
HSI = HSI.astype(np.float32)

DSM = scio.loadmat('Houston2013_Data/Houston2013_DSM.mat')
DSM = DSM['DSM']
DSM = DSM.astype(np.float32)

TR_map = scio.loadmat('Houston2013_Data/Houston2013_TR.mat')
TR_map = TR_map['TR_map']
TE_map = scio.loadmat('Houston2013_Data/Houston2013_TE.mat')
TE_map = TE_map['TE_map']

(m, n, z) = HSI.shape    

for i in range(z):
    ma = np.max(HSI[:, :, i])
    mi = np.min(HSI[:, :, i])
    HSI[:,:,i] = (HSI[:, :, i] - mi)/(ma-mi)   

    ma = np.max(DSM)
    mi = np.min(DSM)
    DSM = (DSM - mi)/(ma-mi)  

    
pad_width = 3
patchsize = 2 * pad_width + 1
k = 0

temp = HSI[:,:,0]
pad_width = np.floor(patchsize/2)
pad_width = np.int(pad_width)
temp2 = np.pad(temp, pad_width, 'symmetric')
[m2,n2] = temp2.shape
HSI2 = np.empty((m2, n2, z), dtype='float32')

for i in range(z):
    temp = HSI[:,:,i]
    pad_width = np.floor(patchsize/2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    HSI2[:,:,i] = temp2
      
temp = DSM
temp2 = np.pad(temp, pad_width, 'symmetric')
DSM2 = temp2
    
temp = TR_map
temp2 = np.pad(temp, pad_width, 'symmetric')
TR_map2 = temp2

temp = TE_map
temp2 = np.pad(temp, pad_width, 'symmetric')
TE_map2 = temp2
#TestPatch = np.zeros(((m-6) * (n-6), patchsize, patchsize, z), dtype='float32')

[ind1_TR, ind2_TR] = np.where(TR_map != 0)
TrainNum = len(ind1_TR)
TrainLabel = np.zeros((TrainNum,1),dtype='uint8')
TrainPatch_HSI = np.zeros((TrainNum, patchsize * patchsize * z), dtype='float32')
TrainPatch_DSM = np.zeros((TrainNum, patchsize * patchsize), dtype='float32')
k_TR = 0

for i in range(3, m2 - 3):
    for j in range(3, n2 - 3):
        patchlabel_TR = TR_map2[i, j]
        if (patchlabel_TR != 0):
            patch_HSI = HSI2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1), :] 
            TrainPatch_HSI[k_TR, :] = np.reshape(patch_HSI, [1, patch_HSI.shape[0] * patch_HSI.shape[1] * patch_HSI.shape[2]], order="F")
            patch_DSM = DSM2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1)] 
            TrainPatch_DSM[k_TR, :] = np.reshape(patch_DSM, [1, patch_DSM.shape[0] * patch_DSM.shape[1]], order="F")
            TrainLabel[k_TR] = patchlabel_TR
            k_TR = k_TR + 1
            
[ind1_TE, ind2_TE] = np.where(TE_map != 0)
TestNum = len(ind1_TE)
TestPatch_HSI = np.zeros((TestNum, patchsize * patchsize * z), dtype='float32')
TestPatch_DSM = np.zeros((TestNum, patchsize * patchsize), dtype='float32')
TestLabel = np.zeros((TestNum,1),dtype='uint8')
k_TE = 0

for i in range(3, m2 - 3):
    for j in range(3, n2 - 3):
        patchlabel_TE = TE_map2[i, j]
        if (patchlabel_TE != 0):
            patch_HSI = HSI2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1), :] 
            TestPatch_HSI[k_TE, :] = np.reshape(patch_HSI, [1, patch_HSI.shape[0] * patch_HSI.shape[1] *patch_HSI.shape[2]], order="F")
            patch_DSM = DSM2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1)] 
            TestPatch_DSM[k_TE, :] = np.reshape(patch_DSM, [1, patch_DSM.shape[0] * patch_DSM.shape[1]], order="F")            
            TestLabel[k_TE] = patchlabel_TE
            k_TE = k_TE + 1
    
Y_train = convert_to_one_hot(TrainLabel-1, 15)
Y_test = convert_to_one_hot(TestLabel-1, 15)

Y_train = Y_train.T
Y_test = Y_test.T

parameters, val_acc, feature = train_mynetwork(TrainPatch_HSI, TrainPatch_DSM, TestPatch_HSI, TestPatch_DSM, Y_train, Y_test)
sio.savemat('feature.mat', {'feature': feature})
sio.savemat('TestLabel.mat', {'TestLabel': TestLabel})
#sio.savemat('feature1.mat', {'feature1': feature1})
print ("Test Accuracy: %f" % (max(val_acc)))  
