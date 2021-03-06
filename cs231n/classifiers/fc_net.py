from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = np.random.normal(scale = weight_scale, size = (input_dim, hidden_dim))
        self.params['b1'] = np.zeros((hidden_dim))
        self.params['W2'] = np.random.normal(scale = weight_scale, size = (hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes) 
       
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        grads = {}
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        H, cache = affine_forward(X, W1, b1) 
        H_actv, cache_relu = relu_forward(H)
        scores, cache_scores = affine_forward(H_actv, W2, b2)
        loss, dloss = softmax_loss(scores, y)        

        if y is None:
            return scores

        
        loss += .5*self.reg*(np.linalg.norm(W1)**2 + np.linalg.norm(W2)**2) 
        dout, grads['W2'], grads['b2'] = affine_backward(dloss, cache_scores) 
        dactv = relu_backward(dout, cache_relu)
        dH, grads['W1'], grads['b1'] = affine_backward(dactv, cache)
        grads['W1'] += self.reg * self.params['W1']
        grads['W2'] += self.reg * self.params['W2'] 
       
        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(self.num_layers):
            w = 'W%d'%(i+1)
            b = 'b%d'%(i+1)
            self.params[w] = np.random.normal(scale = weight_scale, size = (dims[i], dims[i+1])) 
            self.params[b] = np.zeros(dims[i+1])
            if self.use_batchnorm and i < self.num_layers - 1:
                self.params['gamma%d'%(i+1)] = np.ones(dims[i+1])
                self.params['beta%d'%(i+1)] = np.zeros(dims[i+1])
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
           
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            try:
                self.params[k] = v.astype(dtype)
            except: pass

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        h = X
        caches = []
        for i in range(self.num_layers):
            w = self.params['W%d'%(i+1)]
            b = self.params['b%d'%(i+1)]
            if i == self.num_layers - 1:
                scores, cache = affine_forward(h, w, b)
            else: 
                if self.use_batchnorm:
                    gamma = self.params['gamma%d'%(i+1)]
                    beta = self.params['beta%d'%(i+1)]
                    h, cache = affine_bn_relu_forward(h, w, b, gamma, beta, self.bn_params[i], self.use_dropout, self.dropout_param)
                else:
                    h, cache = affine_relu_forward(h, w, b, self.use_dropout, self.dropout_param) 
            caches.append(cache)

        # If test mode return early
        if mode == 'test':
            return scores

        loss, dloss = softmax_loss(scores, y)
        grads = {} 
        dout = dloss
        num_l = self.num_layers
        for i in range(num_l):
            w_key = 'W%d'%(num_l-i)
            b_key = 'b%d'%(num_l-i)
            idx = num_l - i
            if i > 0:
                if self.use_batchnorm:
                    gamma_key = 'gamma%d'%(idx)
                    beta_key = 'beta%d'%(num_l-i)
                    dout, grads[w_key], grads[b_key], grads[gamma_key], grads[beta_key] = affine_bn_relu_backward(dout, caches[idx-1], self.use_dropout) 
                else:
                    dout, grads[w_key], grads[b_key] = affine_relu_backward(dout, caches[idx-1], self.use_dropout)
            else:
                dout, grads[w_key], grads[b_key] = affine_backward(dout, caches[idx-1])

            w = self.params[w_key]
            loss += .5*self.reg*np.linalg.norm(w)**2 
            grads[w_key] += self.reg * w

        return loss, grads
