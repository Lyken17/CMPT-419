import numpy as np
import scipy


def poly_attach(X, n=1):
    """

    Args:
        X: N * M matrix,  N data and each with M features
        n: max degree

    Returns:
        N-degree-X: N * (degree * M) matrix. Which is consisted of [X, X.^2, X.^3 ... X.^n]

    """
    X = np.array(X)
    ans = X
    for i in xrange(2, n+1):
        temp = X ** i
        ans = np.concatenate((temp, ans), axis=1)

    # one line magic in python
    # reduce(lambda a,b: np.concatenate((a,b), axis=1), [np.power(a, i) for i in xrange(1, n+1)])
    return np.matrix(ans)

def polynomial_regression(Xtr, ytr, Xte, yte, degree=1):
    """

    Args:
        Xtr: matrix of training data.   N * M
        ytr: matrix of training answer. N * 1
        Xte: matrix of testing data.    K * M
        yte: matrix of testing answer.  K * 1
        degree: the maximal degree that the data will be augmented.

    Returns:
        W: weights that minimise the least square error. M * 1

    """
    X_tr = poly_attach(Xtr, degree)
    y_tr = ytr
    X_te = poly_attach(Xte, degree)
    y_te = yte
    # print X_tr.shape, y_tr.shape

    w = np.linalg.pinv(X_tr) * y_tr
    err_train = np.sqrt(np.mean(np.square(X_tr * w - y_tr)))
    err_test = np.sqrt(np.mean(np.square(X_te * w - y_te)))

    return w, err_train, err_test

def polynomial_regression_with_regularization(R, Xtr, ytr, Xte, yte, degree=1):
    """

        Args:
            R : the lambda that controls regularization
            Xtr: matrix of training data.   N * M
            ytr: matrix of training answer. N * 1
            Xte: matrix of testing data.    K * M
            yte: matrix of testing answer.  K * 1
            degree: the maximal degree that the data will be augmented.

        Returns:
            W: weights that minimise the least square error. M * 1

    """
    X_tr = poly_attach(Xtr, degree)
    y_tr = ytr
    X_te = poly_attach(Xte, degree)
    y_te = yte
    # print X_tr.shape, y_tr.shape
    temp = X_tr.T * X_tr
    temp += float(R) * np.identity(temp.shape[0])
    w = np.linalg.inv(temp) * X_tr.T * y_tr

    err_train = np.sqrt(np.mean(np.square(X_tr * w - y_tr)))
    err_test = np.sqrt(np.mean(np.square(X_te * w - y_te)))
    return w, err_train, err_test


def sigmoid(t, u, s):
    from math import exp
    y = float(t + u) / s
    return 1.0 / (1.0 + exp(y))


def vectorized_sigmoid(arr, u, s):
    """

    sigmoid = 1 / (1 + exp(-(u + x) / s)
    Args:
        arr: array
        u: center shift
        s: range shrink

    Returns:
        apply sigmoid function to every elements in arr

    """
    vfunc = np.vectorize(sigmoid)
    return vfunc(arr, u, s)


def construct_sigmoid_array(arr, u, s):
    ans = [vectorized_sigmoid(arr, it, s) for it in u]
    return reduce(lambda a,b: np.concatenate((a,b), axis=1), ans)


if __name__ == "__main__":
    k = np.array([[1,2],[3,4]])
    for i in xrange(5):
        print poly_attach(k,i)
