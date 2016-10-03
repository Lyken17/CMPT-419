#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

from utilities import *

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,:]
#x = a1.normalize_data(x)

feature = 10
N_TRAIN = 100;
# Select a single feature.
x_train = x[0:N_TRAIN, feature]
t_train = targets[0:N_TRAIN]
x_test = x[N_TRAIN:, feature]
t_test = targets[N_TRAIN:]

w, err_train, err_test =  polynomial_regression(x_train, t_train, x_test, t_test, degree=3)
# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_test)), np.asscalar(max(x_test)), num=500)
temp = x_ev.reshape(500, 1)
temp = poly_attach(temp, 3)

# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
y_ev = np.random.random_sample(x_ev.shape)
# y_ev = 100 * np.sin(x_ev)
y_ev = temp * w

plt.plot(x_ev,y_ev,'r.-')
plt.plot(x_test,t_test,'bo')
plt.title('A visualization of a testing points and fitting curve of feature {0}'.format(feature+1))
plt.show()
