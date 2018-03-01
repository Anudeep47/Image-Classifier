import numpy as np

## weights and biases initialization

def initialize_parameters(layers_dims):
    parameters = {}
    L = len(layers_dims)-1
    for l in range(1, L+1):
        parameters['W'+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*0.01
        parameters['b'+str(l)] = np.zeros((layers_dims[l], 1))
    return parameters

## utils for forward_propagation

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def relu(Z):
    A = np.maximum(0, Z)
    return A

def linear_activation_forward(A_prev, W, b, activation):
    Z = np.dot(W, A_prev) + b
    if activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'relu':
        A = relu(Z)
    cache = (A_prev, W, b, Z)
    return A, cache

def forward_propagation(X, parameters):
    caches = []
    A = X
    L = len(parameters)//2 #contains W1,b1,W2,b2,...
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)], activation='relu')
        caches.append(cache)
    #for last layer which requires a sigmoid activation
    AL, cache = linear_activation_forward(A, parameters['W'+str(L)], parameters['b'+str(L)], activation='sigmoid')
    caches.append(cache)
    return AL, caches

## computing cost

def compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(np.dot(np.log(AL), Y.T)+np.dot(np.log(1-AL), (1-Y).T))/m
    return np.squeeze(cost)

## utils for backward_propagation

def sigmoid_backward(dA, Z):
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    return dZ

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0]=0
    return dZ

def linear_activation_backward(dA, cache, activation):
    A_prev, W, b, Z = cache
    if activation == 'relu':
        dZ = relu_backward(dA, Z)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, Z)
    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T)/m
    db = np.sum(dZ, axis=1, keepdims=True)/m
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def backward_propagation(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = -np.divide(Y,AL)+np.divide(1-Y,1-AL)
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

## updating parameters

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters)//2
    for l in range(L):
        parameters["W" + str(l+1)] += - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] += - learning_rate * grads["db" + str(l+1)]
    return parameters
