#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def zero_pad(X, pad):
    """
    Zero padding is applied to the height and width of all images of the X. 
    
    Parameters:
    ----------
    X: tensor of shape (m, n_H, n_W, n_C).
        Batch of m images. 
        
    pad: integer
        Amount of padding around each image on vertical and horizontal dims.
    
    Returns:
    X_pad: tensor of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
        Batch of m padded images. 
    """
    
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                   mode = 'constant', constant_values = (0, 0)) 
    
    return X_pad

def conv_single_step(a_slice_prev, W, b):
    """
    Convolve a filter on a slice.
    
    Apply one filter defined by parameters W on a single slice (a_slice_prev)
    of the output activation of the previous layer.
    
    Parameters:
    -----------
    a_slice_prev: tensor of shape (f, f, n_C_prev)
        slice of input data. 
        
    W: tensor of shape (f, f, n_C_prev)
        Weight parameters contained in a window.
        
    b: tensor of shape (1, 1, 1)
        Bias parameters contained in a window.
    
    Returns:
    --------
    Z: float
        The result of convolving the sliding window (W, b) on a slice x of
        the input data.
    """
    
    Z = np.sum(np.multiply(a_slice_prev, W)) + float(b)

    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Parameters:
    -----------
    A_prev: tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        Output activations of previous layer.
        
    W: tensor of shape (f, f, n_C_prev, n_C)
        Weights.
        
    b: tensor of shape (1, 1, 1, n_C)
        Biases.
        
    hparameters: dictionary 
        Contains stride and pad_size.
        
    Returns:
    --------
    A: tensor of shape (m, n_H, n_W, n_C)
        Conv output.
        
    cache: tuple
        cache of values needed for the conv_backward() function.
    """

    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
 
    (f, f, n_C_prev, n_C) = W.shape
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    
    A = np.zeros((m, n_H, n_W, n_C))
    
    A_prev_pad = zero_pad(A_prev, pad)
    
    for i in range(m):               
        a_prev_pad = A_prev_pad[i]
        
        for h in range(n_H):          
            vert_start = h * stride
            vert_end = vert_start + f
            
            for w in range(n_W):                   
                horiz_start = w * stride
                horiz_end = horiz_start + f
                
                for c in range(n_C):   
                    a_slice_prev = a_prev_pad[vert_start:vert_end,
                                              horiz_start:horiz_end, :]  
                    weights = W[:,:,:,c]
                    biases = b[:,:,:,c]
                    A[i, h, w, c] = conv_single_step(a_slice_prev, weights,
                                                     biases)
    
    cache = (A_prev, W, b, hparameters)
    return A, cache

def pool_forward(A_prev, hparameters, mode = "max"):
    """
    Implements the forward pass of the pooling layer.
    
    Parameters:
    -----------
    A_prev: tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        Input data.
        
    hparameters: dictionary 
        Contains "f" and "stride".
        
    mode: str
        Pooling mode you would like to use.
        Options: 
            "max",
            "average".
    
    Returns:
    --------
    A: tensor of shape (m, n_H, n_W, n_C)
        Output of the pool layer.
        
    cache: tuple
        cache used in the backward pass of the pooling layer, 
        contains the input and hparameters .
    """
        
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    f = hparameters["f"]
    stride = hparameters["stride"]
    
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev
    
    A = np.zeros((m, n_H, n_W, n_C))              
    
    for i in range(m):                         
        for h in range(n_H):                    
            vert_start = h * stride
            vert_end = vert_start + f
            
            for w in range(n_W):                 
                horiz_start = w * stride
                horiz_end = horiz_start + f
                
                for c in range (n_C):           
                    
                    a_prev_slice = A_prev[i, vert_start:vert_end,
                                          horiz_start:horiz_end, c]
                    
                    if mode == "max":
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i, h, w, c] = np.mean(a_prev_slice)

    cache = (A_prev, hparameters)
    return A, cache