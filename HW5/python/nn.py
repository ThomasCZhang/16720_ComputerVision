import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    # W, b = None, None

    ##########################
    ##### your code here #####
    variance = 2/(in_size + out_size)
    W = np.random.uniform(-np.sqrt(3*variance),np.sqrt(3*variance), (in_size, out_size))
    b = np.zeros(out_size)
    ##########################

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    # res = None

    ##########################
    ##### your code here #####
    # x = np.clip(x, -50, 50)
    # res[x < 0] = np.exp(x[x < 0])/(1+np.exp(x[x < 0]))
    # res[x >= 0] = 1/(1 + np.exp(-x[x >= 0]))
    res = 1/(1 + np.exp(-x))
    ##########################

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    pre_act = np.matmul(X,W)+b
    post_act = activation(pre_act)
    ##########################


    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    # res = None

    ##########################
    ##### your code here #####
    res = x-np.max(x, axis = 1).reshape(-1, 1) # normalize by max value
    res = np.exp(res)
    res = res/np.sum(res, axis = 1).reshape(-1, 1)
    ##########################

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    # loss, acc = None, None

    ##########################
    ##### your code here #####
    sum = 0
    # loss = 0
    for i, example_prob in enumerate(probs):
        class_idx = np.argmax(example_prob)
        if y[i, class_idx] == 1:
            sum += 1
        # loss -= np.dot(y[i], np.log(probs[i]))
    acc = sum/y.shape[0]
    loss = - np.sum(np.multiply(y, np.log(probs)))
    ##########################

    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    # grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    act_grad = delta*activation_deriv(post_act)
    grad_W = np.matmul(X.T, act_grad)
    grad_b = np.sum(act_grad, axis = 0)
    grad_X = np.matmul(act_grad, W.T)

    ##########################

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    examples = x.shape[0]
    shuffled = np.random.choice(np.arange(examples), examples)
    num_batches = int(np.ceil(examples/batch_size))
    for i in range(num_batches):
        row_ids = shuffled[i*batch_size:(i+1)*batch_size]
        if i == num_batches - 1:
            row_ids = shuffled[i*batch_size:]
        batches.append((x[row_ids], y[row_ids]))
    ##########################
    return batches