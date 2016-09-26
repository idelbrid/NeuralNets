from mlp_theano import TMultiLayerPerceptron
import numpy as np
from sklearn.datasets import fetch_mldata
from FeedForwardNet import MultiLayerPerceptron
import time
import pickle
from theano import function

if __name__ == '__main__':

    def to_distr_repr(y):
        y = y.astype(np.int64)
        yrange = y.max() - y.min()
        toreturn = np.zeros((len(y), int(yrange + 1)))
        for i in range(len(y)):
            toreturn[i][y[i]] = 1
        return toreturn

    def score_1_of_m(model, X, y):  # there is only one correct selection for each datum
        predicted = model.predict_best(X)
        # predicted_class = np.argmax(predicted, axis=1)
        return np.count_nonzero(predicted == y) / len(y)

    SIZES = [784, 800, 300, 100, 10]
    EPOCHS = 100
    REPORT_EVERY = 5
    MINIBATCH_SIZE = 30
    LEARN_RATE = 2
    COST = 'cross entropy'
    REGULARIZER = 'weight decay'
    L = 5.0

    train_size = 0.80
    tune_size = 0.10

    mnist = fetch_mldata('MNIST original', data_home='C:/Users/Ian/Documents/data/')
    mnist['data'] = mnist['data'] / 255.0  # important to normalize the imput to [0,1]
    total_records = mnist['data'].shape[0]

    # MINIBATCH_SIZE = int(total_records * train_size)
    np.random.seed(123456)
    shuffle_indices = np.random.permutation(np.arange(total_records))
    train_indices = shuffle_indices[:int(np.round(train_size * total_records))]
    tune_indices = shuffle_indices[int(np.round(train_size * total_records)):int(np.round((train_size + tune_size) * total_records))]
    test_indices = shuffle_indices[int(np.round((train_size + tune_size) * total_records)):]

    train_X = mnist['data'][train_indices, :]
    tune_X = mnist['data'][tune_indices, :]
    test_X = mnist['data'][test_indices, :]

    train_y_as_num = mnist['target'][train_indices]
    tune_y_as_num = mnist['target'][tune_indices]
    test_y_as_num = mnist['target'][test_indices]

    train_y = to_distr_repr(train_y_as_num)
    tune_y = to_distr_repr(tune_y_as_num)
    test_y = to_distr_repr(test_y_as_num)

    print("Multi-layer perceptron with {} hidden layers of size {}, using {} epochs, {} mini-batch size, and {} "
          "learn-rate".format(len(SIZES)-2, SIZES[1:-1], EPOCHS, MINIBATCH_SIZE, LEARN_RATE))
    print("Using cost function {} and {} regularization (lambda {})".format(COST, REGULARIZER if REGULARIZER else "no",
                                                                           L))
    print("Training Theano model")
    t0 = time.clock()
    model = TMultiLayerPerceptron(SIZES, EPOCHS, MINIBATCH_SIZE, LEARN_RATE, 123456, verbose=1, cost=COST,
                                  report_every=REPORT_EVERY)
    model.fit(train_X, train_y)
    t1 = time.clock()
    print("Elapsed time: {:,} seconds".format(t1 - t0))
    with open('Theano Model.pkl', 'wb') as f:
        pickle.dump(model, f)
    #


    print("Training JITed model")
    t2 = time.clock()
    jmodel = MultiLayerPerceptron(SIZES, EPOCHS, MINIBATCH_SIZE, LEARN_RATE, 123456, verbose=1, cost=COST,
                                  report_every=REPORT_EVERY)
    jmodel.fit(train_X, train_y)
    t3 = time.clock()
    print("Elapsed time: {:,} seconds".format(t3 - t2))
    # with open('JIT Model.pkl', 'wb') as f:
    #     pickle.dump(jmodel, f)

    #
    print('Theano accuracy on train set', score_1_of_m(model, train_X, train_y_as_num))
    print('Theano accuracy on tune set', score_1_of_m(model, tune_X, tune_y_as_num))
    print('Theano accuracy on test set', score_1_of_m(model, test_X, test_y_as_num))
    #
    print('JITed accuracy on train set', score_1_of_m(jmodel, train_X, train_y_as_num))
    print('JITed accuracy on tune set', score_1_of_m(jmodel, tune_X, tune_y_as_num))
    print('JITed accuracy on test set', score_1_of_m(jmodel, test_X, test_y_as_num))

    # print("Theano w0 \n", model.w[0].get_value()[:10, :10])
    # print("JIT w0 \n", jmodel.w[0][:10, :10])
    #
    # print()
    #
    # print("Theano w1 \n", model.w[1].get_value()[:10, :10])
    # print("JIT w1 \n", jmodel.w[1][:10, :10])