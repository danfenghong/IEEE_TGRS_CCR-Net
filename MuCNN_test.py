import numpy as np
#import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as scio 
import scipy.io as sio
from tf_utils import random_mini_batches_standardtwoModality, convert_to_one_hot
from tensorflow.python.framework import ops
from tfdeterminism import patch
patch()

def create_placeholders(n_x1, n_x2):
   
    isTraining = tf.placeholder_with_default(True, shape=())
    x1 = tf.placeholder(tf.float32, [None, n_x1], name = "x1")
    x2 = tf.placeholder(tf.float32, [None, n_x2], name = "x2")
 
    return x1, x2, isTraining


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
            
    with tf.name_scope("encoder_layer_4"):
                
         x1_en_z2 = tf.matmul(x1_en_a1, parameters['x_en_w2']) + parameters['x_en_b2']     
         x1_en_z2_bn = tf.layers.batch_normalization(x1_en_z2, momentum = 0.9, training = isTraining)      
         x1_en_a2 = tf.nn.relu(x1_en_z2_bn)   
         
    with tf.name_scope("encoder_layer_5"):
         
         x1_en_z3 = tf.matmul(x1_en_a2, parameters['x_en_w3']) + parameters['x_en_b3']     
         x1_en_z3_bn = tf.layers.batch_normalization(x1_en_z3, momentum = 0.9, training = isTraining)      
         x1_en_a3 = tf.nn.relu(x1_en_z3_bn)  
         
    with tf.name_scope("decoder_layer_1"):
                
         x1_de_z1 = tf.matmul(x1_en_a3, parameters['x_de_w1']) + parameters['x_de_b1']     
         x1_de_z1_bn = tf.layers.batch_normalization(x1_de_z1, momentum = 0.9, training = isTraining)      
         x1_de_a1 = tf.nn.relu(x1_de_z1_bn)   

    with tf.name_scope("decoder_layer_2"):
         
         x1_de_z2 = tf.matmul(x1_de_a1, parameters['x_de_w2']) + parameters['x_de_b2']     
         x1_de_z2_bn = tf.layers.batch_normalization(x1_de_z2, momentum = 0.9, training = isTraining)      
         x1_de_a2 = tf.nn.relu(x1_de_z2_bn)  

    with tf.name_scope("decoder_layer_3"):
         
         x1_de_z3 = tf.matmul(x1_de_a2, parameters['x_de_w3']) + parameters['x_de_b3']     
         x1_de_z3_bn = tf.layers.batch_normalization(x1_de_z3, momentum = 0.9, training = isTraining)      
         x1_de_a3 = tf.nn.relu(x1_de_z3_bn)  
         
    with tf.name_scope("encoder_layer_6"):
        
         x_en_z4 = tf.matmul(x1_en_a3, parameters['x_en_w4']) + parameters['x_en_b4']

    l2_loss =  tf.nn.l2_loss(parameters['x1_conv_w1']) + tf.nn.l2_loss(parameters['x1_conv_w2']) + tf.nn.l2_loss(parameters['x2_conv_w1']) + tf.nn.l2_loss(parameters['x2_conv_w2'])\
               + tf.nn.l2_loss(parameters['x_en_w1']) + tf.nn.l2_loss(parameters['x_en_w2']) + tf.nn.l2_loss(parameters['x_en_w3']) + tf.nn.l2_loss(parameters['x_en_w4'])\
               + tf.nn.l2_loss(parameters['x_de_w1']) + tf.nn.l2_loss(parameters['x_de_w2']) + tf.nn.l2_loss(parameters['x_de_w3'])
    
    return x_en_z4

def test_mynetwork(x1_test_set, x2_test_set):                
    
    ops.reset_default_graph()                              
    (m1, n_x1) = x1_test_set.shape      
    (m1, n_x2) = x2_test_set.shape     
                 
    x1, x2, isTraining = create_placeholders(n_x1, n_x2)
    parameters = initialize_parameters()
    
    joint_layer = mynetwork(x1, x2, parameters, isTraining)
    # Initialize parameters

    pred_y = np.zeros((m1,15), dtype='float32')
    number = m1 // 5000
    
    saver = tf.train.Saver()
    #init = tf.global_variables_initializer()
    with tf.Session(config=tf.ConfigProto(device_count={'cpu':0})) as sess:    
        #sess.run(init)
        saver.restore(sess, "D:\Python_Project\IGARSS2021/save_MuCNN/model.ckpt")
        #temp2 = sess.run([temp1], feed_dict={x1: x1_test_set, isTraining: False})
        
        for i in range(number):
               temp_1 = x1_test_set[i * 5000 : (i + 1) * 5000, :]
               temp_2 = x2_test_set[i * 5000 : (i + 1) * 5000, :]
               temp2 = sess.run([joint_layer], feed_dict={x1: temp_1, x2: temp_2, isTraining: True})
   
               pred_y[i * 5000 : (i + 1) * 5000, :] = np.reshape(temp2, [5000, 15])
               del temp_1, temp_2, temp2
        
        if m1 - (i + 1) * 5000 < 5000:
               temp_1 = x1_test_set[(i + 1) * 5000: m1, :]
               temp_2 = x2_test_set[(i + 1) * 5000: m1, :]
               temp2 = sess.run([joint_layer], feed_dict={x1: temp_1, x2: temp_2, isTraining: True})
               
               pred_y[(i + 1) * 5000: m1] = np.reshape(temp2, [m1-(i + 1) * 5000, 15])
                
        return pred_y 
        
        
HSI = scio.loadmat('Houston2013_Data/Houston2013_HSI.mat')
HSI = HSI['HSI']
HSI = HSI.astype(np.float32)


DSM = scio.loadmat('Houston2013_Data/Houston2013_DSM.mat')
DSM = DSM['DSM']
DSM = DSM.astype(np.float32)


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

TestPatch1 = np.zeros(((m2-6) * (n2-6), patchsize * patchsize * z), dtype='float32')
TestPatch2 = np.zeros(((m2-6) * (n2-6), patchsize * patchsize * 1), dtype='float32')

for i in range(3, m2 - 3):
    for j in range(3, n2 - 3):
        patch1 = HSI2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1), :] 
        TestPatch1[k, :] = np.reshape(patch1, [1, patch1.shape[0] * patch1.shape[1] *patch1.shape[2]], order="F")
        patch2 = DSM2[(i - pad_width):(i + pad_width + 1), (j - pad_width):(j + pad_width + 1)] 
        TestPatch2[k, :] = np.reshape(patch2, [1, patch2.shape[0] * patch2.shape[1]], order="F")
        k += 1  

all_feature = test_mynetwork(TestPatch1, TestPatch2)
all_feature = np.reshape(all_feature, [m, n, 15])
sio.savemat('all_feature.mat', {'all_feature': all_feature})