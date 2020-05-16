# -*- Coding: utf-8 -*-
"""
File: FCFNN-numpy.py

This Python 3 code is designed to learn DL, only numpy no Tensorflow or Keras.
若在一开始就直接调用框架，小的demo可以跑起来，糊弄一时，看起来就像是鸠摩智在内力未到的情形下强行练习少林寺的72绝技，最后走火入魔。

Author:     Eric Lee
Date:	    2020/5/10
Version     1.0.0
License:    ABC

History:    1. Share project on Github: VCS | Import into Version Control | Share Project on GitHub.
            2. 
"""

import numpy as np


# 激活函数：
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 模型初始化：权值w和偏置b
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    # assert(w.shape == (dim, 1))
    # assert(isinstance(b, float) or isinstance(b, int))
    return w, b


# 定义前向传播函数：
def propagate(w, b, X, Y):
    m = X.shape[1]
    print(m)
    A = sigmoid(np.dot(w.T, X) + b)
    # print(A, w.T, X, b)

    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    grads = {'dw': dw,
             'db': db
             }

    return grads, cost


# 定义反向传播操作：
def backward_propagation(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    cost = []
    for i in range(num_iterations):
        grad, cost = propagate(w, b, X, Y)
        dw = grad['dw']
        db = grad['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        if i % 100 == 0:
            cost.append(cost)
        if print_cost and i % 100 == 0:
            print("cost after iteration %i: %f" % (i, cost))

    params = {"dw": w,
              "db": b
              }
    grads = {"dw": dw,
             "db": db
             }

    return params, grads, cost


# 预测：
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X)+b)
    for i in range(A.shape[1]):
        if A[:, i] > 0.5:
            Y_prediction[:, i] = 1
        else:
            Y_prediction[:, i] = 0
    assert(Y_prediction.shape == (1, m))
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = initialize_with_zeros(X_train.shape[0])    # Gradient descent (≈ 1 line of code)
    # Retrieve parameters w and b from dictionary "parameters"
    print("shape=", X_train.shape[0])
    parameters, grads, costs = backward_propagation(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = parameters["w"]
    b = parameters["b"]    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    return d


if __name__ == '__main__':
    x_data = np.random.randint(low=0, high=10, size=(1000, 3))
    y_data = []

    for i in range(1000):
        y = x_data[i][0]*x_data[i][0] + 2*x_data[i][1] - 0.5*x_data[i][2] + 3
        y_data.append(y)
        # print(x_data[i], y_data)

    # print(x_data[800:1000], y_data[800:1000])
    model(x_data[0:800].T, y_data[0:800], x_data[800:1000].T, y_data[800:1000], num_iterations=2000, learning_rate=0.5, print_cost=False)