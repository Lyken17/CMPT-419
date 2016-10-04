import assignment1 as a1
import numpy as np
import scipy
import matplotlib.pyplot as plt

from utilities import *

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,:]
# x = a1.normalize_data(x)

feature=10
N_TRAIN = 100;
x_train = x[0:N_TRAIN, feature]
t_train = targets[0:N_TRAIN]
x_test = x[N_TRAIN:, feature]
t_test = targets[N_TRAIN:]

u = (100, 10000)
s = 2000.0
r = (0.01, 0.1, 1, 10, 100, 1000, 1000)


new_x_train = construct_sigmoid_array(x_train, u, s)
new_x_test = construct_sigmoid_array(x_test, u, s)


# without regularization
w, err_train, err_test = polynomial_regression(new_x_train, t_train, new_x_test, t_test)
# with regularization
print err_train, err_test
# Plot a curve showing learned function.
# Use linspace to get a set of samples on which to evaluate
x_ev = np.linspace(np.asscalar(min(x_test)), np.asscalar(max(x_test)), num=500)
temp = x_ev.reshape(500, 1)
temp = add_constant(construct_sigmoid_array(temp, u, s))

# TO DO:: Put your regression estimate here in place of x_ev.
# Evaluate regression on the linspace samples.
y_ev = np.random.random_sample(x_ev.shape)
# y_ev = 100 * np.sin(x_ev)
y_ev = temp * w

plt.plot(x_ev,y_ev,'r.-')
plt.plot(x_test,t_test,'bo')
plt.ylim(-1000, 5000)
plt.title('A visualization of a testing points and fitting curve of feature {0}'.format(feature+1))
plt.show()
