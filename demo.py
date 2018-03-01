import numpy as np
import h5py
from model_utils import *

def load_data():
    train_data = h5py.File('data/train_catvnoncat.h5', 'r')
    X_train_data = np.array(train_data['train_set_x'][:])
    Y_train_data = np.array(train_data['train_set_y'][:])

    test_data = h5py.File('data/test_catvnoncat.h5', 'r')
    X_test_data = np.array(test_data['test_set_x'][:])
    Y_test_data = np.array(test_data['test_set_y'][:])

    Y_train_data = Y_train_data.reshape((1,Y_train_data.shape[0]))
    Y_test_data = Y_test_data.reshape((1,Y_test_data.shape[0]))

    classes = np.array(test_data['list_classes'][:])

    return X_train_data, Y_train_data, X_test_data, Y_test_data, classes

def show_data_dims(xtrain, ytrain, xtest, ytest):
    m_train = xtrain.shape[0]
    num_px = xtrain.shape[1]
    m_test = xtest.shape[0]
    print ("Number of training examples: " + str(m_train)) #209
    print ("Number of testing examples: " + str(m_test)) #50
    print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)") #(64, 64, 3)
    print ("xtrain shape: " + str(xtrain.shape)) #(209, 64, 64, 3)
    print ("ytrain shape: " + str(ytrain.shape)) #(1, 209)
    print ("xtest shape: " + str(xtest.shape)) #(50, 64, 64, 3)
    print ("ytest shape: " + str(ytest.shape)) #(1, 50)
    print

def train_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False):
    np.random.seed(1)
    costs = [] #can used for plotting graph
    parameters = initialize_parameters(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = forward_propagation(X, parameters)
        cost = compute_cost(AL, Y)
        grads = backward_propagation(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
    print
    return parameters

def predict(X, y, parameters):
    m = X.shape[1]
    n = len(parameters)//2
    p = np.zeros((1,m))
    probas, caches = forward_propagation(X, parameters)
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    print("Accuracy: "  + str(np.sum((p == y)/m)))
    return p

if __name__ == '__main__':
    X_train, y_train, X_test, y_test, classes = load_data()
    show_data_dims(X_train, y_train, X_test, y_test)

    train_x_flatten = X_train.reshape(X_train.shape[0], -1).T #(12288, 209)
    test_x_flatten = X_test.reshape(X_test.shape[0], -1).T #(12288, 50)

    #standardizing data
    x_train = train_x_flatten/255.
    x_test = test_x_flatten/255.

    layers_dims = [12288, 20, 7, 5, 1]

    parameters = train_model(x_train, y_train, layers_dims, num_iterations=2500, print_cost=True)
    pred_train = predict(x_train, y_train, parameters)
    pred_test = predict(x_test, y_test, parameters)
