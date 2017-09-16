pass
from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b, dropout = False, drop_params = None):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - batchnorm: whether to use batchnorm
    - dropout: whether to use dropout

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    cache = {}
    h, fc_cache = affine_forward(x, w, b)
    cache['fc_cache'] = fc_cache
    h, relu_cache = relu_forward(h)
    cache['relu_cache'] = relu_cache
    if dropout:
        h, drop_cache = dropout_forward(h, drop_params)
        cache['drop_cache'] = drop_cache
    return h, cache


def affine_relu_backward(dout, cache, dropout = False):
    """
    Backward pass for the affine-relu convenience layer
    """
    dx = dout.copy()
    if dropout:
        dx = dropout_backward(dx, cache['drop_cache'])
    dx = relu_backward(dx, cache['relu_cache'])    
    dx, dw, db = affine_backward(dx, cache['fc_cache'])
    return dx, dw, db    
 
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_params, dropout = False, drop_params = None):
    """
    Convenience layer that perorms an affine transform followed by batch normalization and ReLU
    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - gamma, beta: params for batch normalization

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    cache = {}
    h, fc_cache = affine_forward(x, w, b)
    cache['fc_params'] = fc_cache
    h, bn_cache = batchnorm_forward(a, gamma, beta, bn_params)
    cache['bn_cache'] = bn_cache
    h, relu_cache = relu_forward(h)
    cache['relu_cache'] = relu_cache
    if dropout:
        h, drop_cache = dropout_forward(h, drop_params)
        cache['drop_cache'] = drop_cache 
    return h, cache 
   
def affine_bn_relu_backward(dout, cache, dropout = False):
    """
    Backward pass for the affine-batchnorm-relu convenience layer
    """
    dx = dout.copy()
    if dropout:
        dx = dropout_backward(dx, cache['drop_cache']) 
    dx = relu_backward(dx, cache['relu_cache'])
    dx, dgamma, dbeta = batchnorm_backward(dx, cache['bn_cache'])
    dx, dw, db = affine_backward(dbn, cache['fc_params'])
    return dx, dw, db, dgamma, dbeta 

def conv_relu_forward(x, w, b, conv_param):
    """
    A convenience layer that performs a convolution followed by a ReLU.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """
    Backward pass for the conv-relu convenience layer.
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """
    Convenience layer that performs a convolution, a ReLU, and a pool.

    Inputs:
    - x: Input to the convolutional layer
    - w, b, conv_param: Weights and parameters for the convolutional layer
    - pool_param: Parameters for the pooling layer

    Returns a tuple of:
    - out: Output from the pooling layer
    - cache: Object to give to the backward pass
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """
    Backward pass for the conv-relu-pool convenience layer
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db
