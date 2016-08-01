"""
Class wrapper for multi-layer perceptrons using numba and numpy
"""
import numpy as np
import numba as nb
import pickle
from sklearn.datasets import fetch_mldata


@nb.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@nb.jit(nopython=True)
def grad_desc(w, a, y, iterations):
    pass


# u = np.ones((1, x.shape[0]))
@nb.jit(nopython=True)
def sum_2d_ax0(x):
    u = np.ones((1, x.shape[0]))
    return np.dot(u, x)


@nb.jit(nopython=False)
def minibatch_update(X, y, w, b, learn_rate):
    num_layers = len(w) + 1
    a = [None] * num_layers
    a[0] = X
    delta = [None] * (num_layers - 1)
    for i in range(1, num_layers):
        a[i] = sigmoid(np.dot(a[i-1], w[i-1].T) + b[i-1])

    delta[num_layers-2] = (y - a[num_layers-1]) * (a[num_layers-1] * (1 - a[num_layers -1]))
    for i in range(num_layers - 2, 0, -1):
        delta[i-1] = np.dot(delta[i], w[i]) * (a[i] * (1 - a[i]))

    for i in range(num_layers - 1):
        w[i] += learn_rate * np.dot(delta[i].T, a[i]) / len(X)
        b[i] += learn_rate * sum_2d_ax0(delta[i]) / len(X)
    return w, b


@nb.jit(nopython=False)
def run_minibatches(X, y, w, b, learn_rate, batch_size):
    for i in range(int(np.ceil(len(y) / batch_size))):
        w, b = minibatch_update(X[(i*batch_size):((i+1)*batch_size), :],
                                y[(i*batch_size):((i+1)*batch_size)],
                                w, b, learn_rate)
    return w, b


class MultiLayerPerceptron:
    def __init__(self, layer_sizes, epochs, batch_size, learn_rate, init_seed=None, verbose=False):
        assert(len(layer_sizes) > 1)
        np.random.seed(init_seed)
        w = [None] * (len(layer_sizes) - 1)
        b = [None] * (len(layer_sizes) - 1)
        for i in range(len(layer_sizes)-1):
            w[i] = np.random.randn(layer_sizes[i+1], layer_sizes[i])
            b[i] = np.random.randn(1, layer_sizes[i+1])

        self.init_seed = init_seed
        self.num_layers = len(layer_sizes)
        self.w = w
        self.b = b
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.layer_sizes = layer_sizes
        self.verbose = verbose
        # self.y = None

    def feed_forward(self, X):
        a = X
        for i in range(self.num_layers-1):
            a = sigmoid(np.dot(a, self.w[i].T) + self.b[i])
        return a

    def run_epoch(self, X, y):
        if self.verbose:
            score = np.abs(y - self.feed_forward(X)).mean()

        shuffle_indices = np.arange(len(X))
        np.random.shuffle(shuffle_indices)
        X = X[shuffle_indices, :]
        y = y[shuffle_indices, :]
        self.w, self.b = run_minibatches(X, y, self.w.copy(), self.b.copy(), self.learn_rate, self.batch_size)

    def fit(self, X, y):
        assert(X.shape[0] == y.shape[0])
        assert(X.shape[1] == self.w[0].shape[1])
        assert(y.shape[1] == self.w[self.num_layers-2].shape[0])

        for ep in range(self.epochs):
            self.run_epoch(X, y)

    def predict(self, X):  # alias for feed-forward
        return self.feed_forward(X)

    def score_1_of_m(self, X, y):  # there is only one correct selection for each datum
        predicted = self.predict(X)
        predicted_class = np.argmax(predicted, axis=1)
        return np.count_nonzero(predicted_class == y) / len(y)

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
    SIZES = [784, 100, 30, 10]
    EPOCHS = 70
    MINIBATCH_SIZE = 20
    LEARN_RATE = 5

    train_size = .80

    mnist = fetch_mldata('MNIST original', data_home='C:/Users/Ian/Documents/data/')
    mnist['data'] = mnist['data'] / 255.0  # important to normalize the imput to [0,1]
    total_records = mnist['data'].shape[0]

    shuffle_indices = np.random.permutation(np.arange(total_records))
    train_indices = shuffle_indices[:train_size*total_records]
    test_indices = shuffle_indices[train_size*total_records:]

    train_X = mnist['data'][train_indices, :]
    test_X = mnist['data'][test_indices, :]

    train_y_as_num = mnist['target'][train_indices]
    test_y_as_num = mnist['target'][test_indices]

    train_y = to_distr_repr(train_y_as_num)
    test_y = to_distr_repr(test_y_as_num)

    print("Multi-layer perceptron with {} hidden layers of size {}, using {} epochs, {} mini-batch size, and {} "
          "learn-rate".format(len(SIZES), SIZES, EPOCHS, MINIBATCH_SIZE, LEARN_RATE))
    model = MultiLayerPerceptron(SIZES, EPOCHS, MINIBATCH_SIZE, LEARN_RATE, 123456)
    model.fit(train_X, train_y)
    model.save('MLP class.pkl')
    # model = MultiLayerPerceptron.load('MLP class.pkl')

    print('Accuracy on train set', model.score_1_of_m(train_X, train_y_as_num))
    print('Accuracy on test set', model.score_1_of_m(test_X, test_y_as_num))








