"""
Class wrapper for multi-layer perceptrons using numba and numpy
"""
import numpy as np
from numba import jit, typeof
import pickle

@jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@jit(nopython=True)
def grad_desc(w, a, y, iterations):
    pass


# u = np.ones((1, x.shape[0]))
@jit(nopython=True)
def sum_2d_ax0(x):
    u = np.ones((1, x.shape[0]))
    return np.dot(u, x)


class MultiLayerPerceptron:
    def __init__(self, layer_sizes, epochs, batch_size, learn_rate, init_seed=None):
        assert(len(layer_sizes) > 1)
        np.random.seed(init_seed)
        w = np.zeros(len(layer_sizes) - 1, dtype=object)
        b = np.zeros(len(layer_sizes) - 1, dtype=object)
        a = np.zeros(len(layer_sizes), dtype=object)
        for i in range(len(layer_sizes)-1):
            w[i] = np.random.randn(layer_sizes[i], layer_sizes[i+1])
            b[i] = np.random.randn(1, layer_sizes[i+1])
        for i in range(len(layer_sizes)):
            a[i] = np.zeros((layer_sizes[i]))

        self.init_seed = init_seed
        self.num_layers = len(layer_sizes)
        self.w = w
        self.b = b
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.layer_sizes = layer_sizes
        # self.y = None

    def feed_forward(self, X):
        a = X
        for i in range(self.num_layers-1):
            a = sigmoid(np.dot(a, self.w[i].T) + self.b[i])
        return a

    def minibatch_update(self, X, y):
        a = np.zeros(self.num_layers, dtype=object)
        delta = np.zeros(self.num_layers - 1, dtype=object)
        w = self.w
        b = self.b
        for i in range(1, self.num_layers):
            a[i] = sigmoid(np.dot(a[i-1], w[i-1].T + self.b[i-1]))

        delta[self.num_layers-2] = (y - a[self.num_layers-1]) * (a[self.num_layers-1] * (1 - a[self.num_layers -1]))
        for i in range(self.num_layers - 2, 0):
            delta[i-1] = np.dot(delta[i], w[i])

        for i in range(self.num_layers - 1):
            w[i] += self.learn_rate * np.dot(delta[i].T, a[i]) / len(X)
            b[i] += self.learn_rate * sum_2d_ax0(delta[i]) / len(X)
        self.w = w
        self.b = b

    def run_epoch(self, X, y):
        shuffle_indices = np.arange(len(X))
        np.random.shuffle(shuffle_indices)
        X = X[shuffle_indices, :]
        y = y[shuffle_indices, :]
        for i in range(int(np.ceil(len(y) / self.batch_size))):
            # used_range = (i*self.batch_size):((i+1)*self.batch_size)
            self.minibatch_update(X[(i*self.batch_size):((i+1)*self.batch_size), :],
                                  y[(i*self.batch_size):((i+1)*self.batch_size)])

    def fit(self, X, y):
        assert(X.shape[0] == y.shape[0])
        assert(X.shape[1] == self.w[0].shape[0])
        assert(y.shape[1] == self.w[self.num_layers-1].shape[1])
        # a = np.zeros(self.num_layers, dtype=object)
        # a[0] = X
        #
        for ep in range(self.epochs):
            self.run_epoch(X, y)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickler = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
            components = {'layer_sizes': self.layer_sizes,
                          'init_seed': self.init_seed,
                          'weights': self.w,
                          'biases': self.b,
                          'num_layers': self.num_layers,
                          'batch_size': self.batch_size,
                          'learn_rate': self.learn_rate,
                          'num_epochs': self.epochs}
            pickler.dump(components)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as f:
            components = pickle.load(f)
            sizes = components['layer_sizes']
            seed = components['init_seed']
            w = components['weights']
            b = components['biases']
            # num_layers = components['num_layers']
            batch_size = components['batch_size']
            learn_rate = components['learn_rate']
            epochs = components['num_epochs']
            model = MultiLayerPerceptron(layer_sizes=sizes, epochs=epochs, batch_size=batch_size,
                                         learn_rate=learn_rate, init_seed=seed)
            model.w = w
            model.b = b


def to_distr_repr(y):
    yrange = y.max() - y.min()
    toreturn = np.zeros((len(y), int(yrange+1)))
    for i in range(len(y)):
        toreturn[i][y[i]] = 1
    return toreturn


if __name__ == '__main__':
    




