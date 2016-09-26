from theano import tensor as T, function, shared
import numpy as np


class TMultiLayerPerceptron(object):
    def __init__(self, layer_sizes, epochs, batch_size, learn_rate, init_seed=None, verbose=False, cost='MSE',
                 regularizer=None, l=1, report_every=1):
        """
        A multi-layer perceptron implemented using Theano
        :param layer_sizes: sizes of the layers, including input and output
        :param epochs: number of epochs to run
        :param batch_size: number of samples in each minibatch.
        :param learn_rate: hyperparameter controlling learning velocity
        :param init_seed: used to seed numpy RNG
        :param verbose: 0 for nothing, 1 for cost updates
        :param cost: 'MSE'(mean squared error), 'cross entropy'
        :param regularizer: 'weight decay' or None
        :param l: coefficient of regularizer
        """

        assert len(layer_sizes) > 1
        self.init_seed = init_seed
        self.layer_sizes = layer_sizes
        self.epochs = epochs
        self.batch_size = batch_size
        self.learn_rate = learn_rate
        self.verbose = verbose
        self.cost = cost
        self.regularizer = regularizer
        self.l = l
        self.cost_value = None
        self.report_every = report_every

        n_layers = len(layer_sizes)
        trainxvar, trainyvar = T.dmatrices('xt', 'yt')
        x, y = T.dmatrices('x', 'y')

        # initializing the weights and biases randomly
        weights = []
        biases = []
        np.random.seed(init_seed)
        for i in range(n_layers-1):
            weights.append(shared(np.random.randn(layer_sizes[i], layer_sizes[i+1]), name='w{}'.format(i)))
            biases.append(shared(np.random.randn(layer_sizes[i+1]), name='b{}'.format(i)))

        # forward propagation
        a = []
        for i in range(n_layers-1):
            if i == 0:
                a.append(
                    1 / (1 + T.exp(-(T.dot(x, weights[i]) + biases[i])))
                )
            else:
                a.append(
                    1 / (1 + T.exp(-(T.dot(a[i-1], weights[i]) + biases[i])))
                )
        self.a = a
        self.w = weights

        self._feed_forward = function([x], a[-1])
        self._predict_best = function([x], a[-1].argmax(axis=1))
        self._predict_activation = function([x], a[-1].round())

        # creating cost function
        if cost == 'MSE':
            err = (y - a[-1]) ** 2 / 2
        elif cost == 'cross entropy':
            err = -y * T.log(a[-1]) - (1 - y)*T.log(1 - a[-1])
        else:
            raise ValueError("Unknown cost function, {}".format(cost))

        # adding regularization function to it
        if regularizer is None:
            cost_f = err.mean()
        elif regularizer == 'weight decay':
            cost_f = err.mean() + l / (2 * x.size[0]) * T.sum([(_w ** 2).sum() for _w in weights])
        else:
            raise ValueError("Unknown regularization method, {}".format(regularizer))

        self._get_cost = function([x, y], cost_f)

        # creating training function
        dweights = T.grad(cost_f, weights)
        dbiases = T.grad(cost_f, biases)
        idx = T.lscalar()
        self._train = function(inputs=[idx, trainxvar, trainyvar],
                               outputs=[cost_f],
                               updates=[(_w, _w - learn_rate * _gw) for _w, _gw in zip(weights, dweights)] +
                                       [(_b, _b - learn_rate * _gb) for _b, _gb in zip(biases, dbiases)],
                               givens=[
                                   (x, trainxvar[batch_size*idx: batch_size*(idx+1)]),
                                   (y, trainyvar[batch_size*idx: batch_size*(idx+1)])
                               ])

    def fit(self, X, y):
        np.random.seed(self.init_seed)
        shuffle_indices = np.arange(len(X))

        for epoch in range(self.epochs):
            np.random.shuffle(shuffle_indices)
            X = X[shuffle_indices, :]
            y = y[shuffle_indices, :]
            if self.verbose:
                if epoch % self.report_every == 0:
                    cost = self._get_cost(X, y)
                    print("Epoch {}: cost = {:2.6f}".format(epoch, float(cost)))

            for i, idx in enumerate(range(0, len(X), self.batch_size)):
                cost = self._train(i, X, y)

    def feed_forward(self, X):
        return self._feed_forward(X)

    def predict_best(self, X):
        return self._predict_best(X)

    def predict_activation(self, X):
        return self._predict_activation

    def get_cost(self):
        return self.cost_value
