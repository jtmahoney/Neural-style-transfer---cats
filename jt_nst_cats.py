# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 13:49:21 2017

@author: remote
"""
#In[1]:

import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from jt_nst_cats_utils import *
import numpy as np
import tensorflow as tf

imgs = ['bear2.jpg']
styls = ['art15.jpg']

get_ipython().magic('matplotlib inline')
model = load_vgg_model("imagenet-vgg-verydeep-19.mat")
print(model)
#content_image = scipy.misc.imread(im)
#style_image = scipy.misc.imread(sty)
"""
Change IM_SIZE to change the size of the outut imgae.  Larger images take exponentially larger ammounts of time.

Also change the sizes in jt_nst_cats_utils.py
"""
IM_SIZE = [700,900]

STYLE_LAYERS = [
    ('conv1_1', 0.5),
    ('conv2_1', 1.0),
    ('conv3_1', 1.5),
    ('conv4_1', 3.0),
    ('conv5_1', 4.0)]

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_C_unrolled = tf.reshape(a_C,[n_C, n_H*n_W])
    a_G_unrolled = tf.reshape(a_G,[n_C, n_H*n_W])
    J_content = (1/(4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    
    return J_content


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    GA = tf.matmul(A,tf.transpose(A))
    return GA



def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    a_S = tf.reshape(a_S,[n_H*n_W, n_C])
    a_G = tf.reshape(a_G,[n_H*n_W, n_C])
    GS = gram_matrix(tf.transpose(a_S))
    GG = gram_matrix(tf.transpose(a_G))
    J_style_layer = (1/(4*((n_C)**2)*((n_H*n_W)**2)))*(tf.reduce_sum(tf.reduce_sum(tf.square(tf.subtract(GS, GG)),1),0))
    
    return J_style_layer


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:
        out = model[layer_name]
        a_S = sess.run(out)
        a_G = out
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style




def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha*J_content + beta*J_style
    return J



def model_nn(sess, input_image, input_image_name, style_image_name, num_iterations = 1000
             ):
    
    sess.run(tf.global_variables_initializer())

    sess.run(model['input'].assign(input_image))
    
    for i in range(num_iterations):
    
        sess.run(train_step)
    
        generated_image = sess.run(model['input'])
        
        if i%20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            # save current generated image in the "/output" directory
            save_image("output/" + str(input_image_name) + '_with_' + str(style_image_name) + str(i) + ".png", generated_image)
    
    # save last generated image
    save_image('output/' + str(input_image_name) + '_with_' + str(style_image_name) + '_generated_image.jpg', generated_image)
    
    return generated_image



for im in imgs:
    im_path = 'images/' + str(im)
    for sty in styls:
        sty_path = 'images/' + str(sty)

        tf.reset_default_graph()
        sess = tf.InteractiveSession()


        content_image = scipy.misc.imread(im_path)
        content_image = scipy.misc.imresize(content_image, size=IM_SIZE)
        content_image = reshape_and_normalize_image(content_image)

        style_image = scipy.misc.imread(sty_path)
        style_image = scipy.misc.imresize(style_image, size=IM_SIZE)
        
        style_image = reshape_and_normalize_image(style_image)
        #imshow(style_image[0])

        generated_image = generate_noise_image(content_image)
        model = load_vgg_model("imagenet-vgg-verydeep-19.mat")
        sess.run(model['input'].assign(content_image))


        out = model['conv4_2']

        a_C = sess.run(out)

        a_G = out

        J_content = compute_content_cost(a_C, a_G)

        sess.run(model['input'].assign(style_image))

        J_style = compute_style_cost(model, STYLE_LAYERS)

        J = total_cost(J_content, J_style, alpha = 10, beta = 40)

        optimizer = tf.train.AdamOptimizer(2.0)

        train_step = optimizer.minimize(J)

        model_nn(sess, generated_image, im, sty)