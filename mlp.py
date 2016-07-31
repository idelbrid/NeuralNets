"""
Class wrapper for multi-layer perceptrons using numba and numpy
"""
import numpy as np
from numba import jit, typeof


@jit(typeof(np.zeros((2, 2), dtype=np.float64))(typeof(np.zeros((2, 2)))))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@jit
def grad_desc(w, a, y, iterations):

    for i in iterations:
        pass





class MultiLayerPerceptron:
    def __init__(self, layer_sizes, iterations, init_seed=None):
        assert(len(layer_sizes) > 1)
        np.random.seed(init_seed)
        w = np.zeros(len(layer_sizes) -1, dtype=object)
        a = np.zeros(len(layer_sizes), dtype=object)
        for i in range(len(layer_sizes)-1):
            w[i] = 2*np.random.random((layer_sizes[i], layer_sizes[i+1])) - 1
        for i in range(len(layer_sizes)):
            a[i] = np.zeros((layer_sizes[i]))

        self.num_layers = len(layer_sizes)
        self.w = w
        self.a = a
        self.iterations = iterations
        self.y = None

    def fit(self, X, y):
        assert(X.shape[1] == self.w[0].shape[0])
        assert(y.shape[1] == len(self.a[self.num_layers-1]))
        self.a[0] = X
        self.y = y

        grad_desc()


