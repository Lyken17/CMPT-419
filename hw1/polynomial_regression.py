#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import scipy
import matplotlib.pyplot as plt

from utilities import polynomial_regression

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
# x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_err = dict()
test_err = dict()
for i in xrange(1, 7):
    w, err_train, err_test = polynomial_regression(x_train, t_train, x_test, t_test, degree=i)
    train_err[i] = err_train
    test_err[i] = err_test

plt.plot(test_err.keys(), test_err.values())
plt.plot(train_err.keys(), train_err.values())
plt.ylabel('RMS')
plt.legend(['Test error', 'Training error'])
plt.title('Fit with polynomials, without normalization')
plt.xlabel('Polynomial degree')
plt.show()


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
x = a1.normalize_data(x)


N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_err = dict()
test_err = dict()
for i in xrange(1, 7):
    w, err_train, err_test = polynomial_regression(x_train, t_train, x_test,t_test, degree=i)
    train_err[i] = err_train
    test_err[i] = err_test

plt.plot(test_err.keys(), test_err.values())
plt.plot(train_err.keys(), train_err.values())
plt.ylabel('RMS')
plt.legend(['Test error', 'Training error'])
plt.title('Fit with polynomials, with normalization')
plt.xlabel('Polynomial degree')
plt.show()
