"""
Class wrapper for multi-layer perceptrons using numba and numpy
"""
import numpy as np
import numba as nb
import pickle
from sklearn.datasets import fetch_mldata

# ToDO: add docstrings

@nb.jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@nb.jit(nopython=True)
def sum_2d_ax0(x):
    u = np.ones((1, x.shape[0]))
    return np.dot(u, x)


@nb.jit(nopython=False)
def minibatch_update(X, y, w, b, learn_rate, top_layer_delta, regularizer, l):
    # k = np.float64_t
    num_layers = len(w) + 1
    a = [None] * num_layers
    a[0] = X
    delta = [None] * (num_layers - 1)
    for i in range(1, num_layers):
        a[i] = sigmoid(np.dot(a[i-1], w[i-1].T) + b[i-1])

    delta[num_layers-2] = top_layer_delta(y, a[num_layers-1])
    for i in range(num_layers - 2, 0, -1):
        delta[i-1] = np.dot(delta[i], w[i]) * (a[i] * (1 - a[i]))

    for i in range(num_layers - 1):
        w[i] += learn_rate * np.dot(delta[i].T, a[i]) / len(X) - learn_rate * regularizer(w[i], len(X), l)
        b[i] += learn_rate * sum_2d_ax0(delta[i]) / len(X)
    return w, b


@nb.jit(nopython=False)
def run_minibatches(X, y, w, b, learn_rate, batch_size, top_layer_delta, regularizer, l):
    for i in range(int(np.ceil(len(y) / batch_size))):
        w, b = minibatch_update(X[(i*batch_size):((i+1)*batch_size), :],
                                y[(i*batch_size):((i+1)*batch_size)],
                                w, b, learn_rate, top_layer_delta,
                                regularizer, l)
    return w, b


@nb.jit
def MSE_update(y, a):
    return (y - a) * (a * (1 - a))


@nb.jit
def cross_entropy_update(y, a):
    return y - a


@nb.jit
def cross_entropy(y, a):
    return -y * np.log(a) - (1 - y) * np.log(1 - a)


@nb.jit
def MSE(y, a):
    return (y - a) ** 2 / 2


@nb.jit
def L2Regularizer(w, n, l):
    return (w ** 2).sum() * l / (2 * n)


@nb.jit
def L1Regularizer(w, n, l):
    return (np.abs(w)).sum() * l / n


@nb.jit
def NoRegularizer(w, n, l):
    return 0


@nb.jit
def rglzr_none_update(w, n, l):
    return 0


@nb.jit
def rglzr_wd_update(w, n, l):
    return l * w / n


@nb.jit
def rglzr_l1_update(w, n, l):
    return l / n


class MultiLayerPerceptron:
    def __init__(self, layer_sizes, epochs, batch_size, learn_rate, init_seed=None, verbose=False, cost='MSE',
                 regularizer=None, l=1, report_every=1):
        assert(len(layer_sizes) > 1)
        np.random.seed(init_seed)
        w = [None] * (len(layer_sizes) - 1)
        b = [None] * (len(layer_sizes) - 1)
        for i in range(len(layer_sizes)-1):
            w[i] = np.random.randn(layer_sizes[i+1], layer_sizes[i])
            b[i] = np.random.randn(1, layer_sizes[i+1])

        if cost == 'MSE':
            self.cost_fn_update = MSE_update
            self.cost = MSE
        elif cost == 'cross entropy':
            self.cost_fn_update = cross_entropy_update
            self.cost = cross_entropy
        else:
            raise ValueError("Not a valid cost function")

        if regularizer is None:
            self.regularizer_update = rglzr_none_update
            self.regularizer = NoRegularizer
        elif regularizer == 'weight decay':
            self.regularizer_update = rglzr_wd_update
            self.regularizer = L2Regularizer
        elif cost == 'L1':
            self.regularizer_update = rglzr_l1_update
            self.regularizer = L1Regularizer
        else:
            raise ValueError("Not a valid regularizer")

        self.init_seed = init_seed
        self.num_layers = len(layer_sizes)
        self.w = w
        self.b = b
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.epochs = epochs
        self.layer_sizes = layer_sizes
        self.verbose = verbose
        self.l = l
        self.report_every = report_every
        # self.y = None

    def feed_forward(self, X):
        a = X
        for i in range(self.num_layers-1):
            a = sigmoid(np.dot(a, self.w[i].T) + self.b[i])
        return a

    def run_epoch(self, X, y, shuffle_indices):  # run a single epoch

        np.random.shuffle(shuffle_indices)
        X = X[shuffle_indices, :]
        y = y[shuffle_indices, :]
        self.w, self.b = run_minibatches(X, y, self.w.copy(), self.b.copy(), self.learn_rate, self.batch_size,
                                         self.cost_fn_update, self.regularizer_update, self.l)

    def fit(self, X, y):
        assert(X.shape[0] == y.shape[0])
        assert(X.shape[1] == self.w[0].shape[1])
        assert(y.shape[1] == self.w[self.num_layers-2].shape[0])
        np.random.seed(self.init_seed)
        shuffle_indices = np.arange(len(X))
        for ep in range(self.epochs):
            if self.verbose:
                if ep % self.report_every == 0:
                    cost = self.get_cost(X, y)
                    print("Epoch {}: cost = {:2.6f}".format(ep, cost))
            self.run_epoch(X, y, shuffle_indices=shuffle_indices)

    def predict_best(self, X):
        return self.feed_forward(X).argmax(axis=1)

    def predict_activate(self, X):
        return self.feed_forward(X).round()

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

    def get_cost(self, X, y):
        a = self.feed_forward(X)
        total_reg_cost = 0
        for _w in self.w:
            total_reg_cost += self.regularizer(_w, len(X), self.l)
        return self.cost(y, a).mean() + total_reg_cost

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
        return model


def to_distr_repr(y):
    yrange = y.max() - y.min()
    toreturn = np.zeros((len(y), int(yrange+1)))
    for i in range(len(y)):
        toreturn[i][y[i]] = 1
    return toreturn


def score_1_of_m(model, X, y):  # there is only one correct selection for each datum
    predicted = model.predict_best(X)
    # predicted_class = np.argmax(predicted, axis=1)
    return np.count_nonzero(predicted == y) / len(y)


#
# if __name__ == '__main__':
#     SIZES = [784, 800, 10]
#     EPOCHS = 5
#     MINIBATCH_SIZE = 15
#     LEARN_RATE = 2
#     COST = 'cross entropy'
#     REGULARIZER = "weight decay"
#     L = 5.0
#
#     train_size = 0.80
#     tune_size = 0.10
#
#     mnist = fetch_mldata('MNIST original', data_home='C:/Users/Ian/Documents/data/')
#     mnist['data'] = mnist['data'] / 255.0  # important to normalize the imput to [0,1]
#     total_records = mnist['data'].shape[0]
#     np.random.seed(123456)
#     shuffle_indices = np.random.permutation(np.arange(total_records))
#     train_indices = shuffle_indices[:train_size*total_records]
#     tune_indices = shuffle_indices[train_size*total_records:(train_size+tune_size)*total_records]
#     test_indices = shuffle_indices[(train_size+tune_size)*total_records:]
#
#     train_X = mnist['data'][train_indices, :]
#     tune_X = mnist['data'][tune_indices, :]
#     test_X = mnist['data'][test_indices, :]
#
#     train_y_as_num = mnist['target'][train_indices]
#     tune_y_as_num = mnist['target'][tune_indices]
#     test_y_as_num = mnist['target'][test_indices]
#
#     train_y = to_distr_repr(train_y_as_num)
#     tune_y = to_distr_repr(tune_y_as_num)
#     test_y = to_distr_repr(test_y_as_num)
#
#     print("Multi-layer perceptron with {} hidden layers of size {}, using {} epochs, {} mini-batch size, and {} "
#           "learn-rate".format(len(SIZES)-2, SIZES[1:-1], EPOCHS, MINIBATCH_SIZE, LEARN_RATE))
#     print("Using cost function {} and {} regularization (lamba {})".format(COST, REGULARIZER if REGULARIZER else "no",
#                                                                            L))
#
#     model = MultiLayerPerceptron(SIZES, EPOCHS, MINIBATCH_SIZE, LEARN_RATE, 123456, verbose=True, cost=COST)
#     model.fit(train_X, train_y)
#     model.save('MLP class.pkl')
#     # model = MultiLayerPerceptron.load('MLP class.pkl')
#
#     print('Accuracy on train set', score_1_of_m(model, train_X, train_y_as_num))
#     print('Accuracy on tune set', score_1_of_m(model, tune_X, tune_y_as_num))
#     print('Accuracy on test set', score_1_of_m(model, test_X, test_y_as_num))

    # import matplotlib.pyplot as plt
    # sample_test_x = test_X[:20]
    # sample_test_y_num = test_y_as_num[:20]
    # sample_pred_test_y_num = np.argmax(model.predict(test_X[:20]), axis=1)
    #
    # for x, y, pred in zip(sample_test_x, sample_test_y_num, sample_pred_test_y_num):
    #     plt.pcolor(np.reshape(x, (28, 28))[::-1, ::1], cmap='Greys')
    #     plt.title('Computer says it\'s a {}'.format(pred))
    #     plt.show()








