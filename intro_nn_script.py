"""
Making and training a basic neural net on the MNIST dataset

"""
import numpy as np
import numba as nb
from sklearn.datasets import fetch_mldata
import pickle

ex_np_arr = np.zeros((2, 2), dtype=np.float64)
ar_type = nb.typeof(ex_np_arr)


@nb.jit(ar_type(ar_type), nopython=True)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# @nb.jit(nb.typeof((ex_np_arr, ex_np_arr))(ar_type, ar_type, ar_type, ar_type, nb.typeof(6000)),
#         nopython=True)
@nb.jit
def train(X, y, w0, w1, iterations, batch_size=500, learn_rate=3.0):

    bs = batch_size
    # Xpartition = [X[bs*i:bs*(i+1)] for i in range(int(np.ceil(len(X)/bs)))]
    # ypartition = [y[bs*i:bs*(i+1)] for i in range(int(np.ceil(len(y)/bs)))]
    # best_w0, best_w1 = None, None
    # best_score = np.inf
    for it in range(iterations):
        # cur_score = np.abs(y - sigmoid(np.dot(sigmoid(np.dot(X, w0.T)), w1.T))).sum()
        # print('Epoch', it, 'Error ', cur_score)
        # if cur_score < best_score:
        #     best_score = cur_score
        #     best_w0 = w0.copy()
        #     best_w1 = w1.copy()

        shuffle_indices = np.random.permutation(np.arange(len(X)))
        X = X[shuffle_indices, :]
        y = y[shuffle_indices, :]

        for i in range(int(np.ceil(len(y) / bs))):
            a0 = X[(i*bs):((i+1)*bs)]
            t = y[(i*bs):((i+1)*bs)]
            a1 = sigmoid(np.dot(a0, w0.T))
            a2 = sigmoid(np.dot(a1, w1.T))

            delta2 = (t - a2) * (a2 * (1 - a2))
            delta1 = np.dot(delta2, w1) * (a1 * (1 - a1))

            w1 += learn_rate * np.dot(delta2.T, a1) / bs
            w0 += learn_rate * np.dot(delta1.T, a0) / bs

    return (w0, w1)


@nb.jit(ar_type(ar_type, ar_type, ar_type), nopython=True)
def feed_forward(X, w0, w1):
    return sigmoid(np.dot(sigmoid(np.dot(X, w0.T)), w1.T))  # need to add biases


def to_distr_repr(y):
    yrange = y.max() - y.min()
    toreturn = np.zeros((len(y), int(yrange+1)))
    for i in range(len(y)):
        toreturn[i][y[i]] = 1
    return toreturn


def save_model(w0, w1, filename='./mlp_model.pkl'):

    with open(filename, 'wb') as f:
        pickler = pickle.Pickler(f, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump((w0, w1))


def load_model(filename='./mlp_model.pkl'):

    with open(filename, 'rb') as f:
        (w0, w1) = pickle.load(f)
    return w0, w1


if __name__ == '__main__':

    # hidden_layer_size = 2
    # X = np.array([
    #     [0, 1, 1, 0],
    #     [0, 0, 0, 0],
    #     [1, 0, 0, 1],
    #     [0, 0, 0, 1],
    #     [0, 1, 0, 1]
    # ])
    # y = np.array([[1, 0, 1, 0, 1]]).T
    # myw0 = 2 * np.random.random((hidden_layer_size, X.shape[1])) - 1
    # myw1 = 2 * np.random.random((1, hidden_layer_size))
    #
    # w0s = []
    # w1s = []
    # w0, w1 = train(X, y, myw0, myw1, 2000, batch_size=2)
    #
    # print("Stop debugger here")

    hidden_layer_size = 30
    np.random.seed(123456)
    train_size = 0.80

    mnist = fetch_mldata('MNIST original', data_home='C:/Users/Ian/Documents/data/')
    mnist['data'] = mnist['data'] / 255.0
    total_records = mnist['data'].shape[0]

    shuffled_indices = np.random.permutation(np.arange(total_records))
    train_indices = shuffled_indices[:train_size*total_records]
    test_indices = shuffled_indices[train_size*total_records:]

    X = mnist['data'][train_indices, :]
    test_X = mnist['data'][test_indices, :]
    y_numbers = mnist['target'][train_indices]
    y = to_distr_repr(y_numbers)

    test_y_numbers = mnist['target'][test_indices]
    test_y = to_distr_repr(test_y_numbers)

    # myw0 = 2 * np.random.random((hidden_layer_size, X.shape[1])) - 1
    # myw1 = 2 * np.random.random((10, hidden_layer_size)) - 1

    myw0 = np.random.randn(hidden_layer_size, X.shape[1])
    myw1 = np.random.randn(10, hidden_layer_size)
    # myb1 = np.random.randn(1, hidden_layer_size)
    # myb2 = np.random.randn(1, 10)

    # w0, w1 = train(X, y, myw0, myw1, 150, 15)
    # save_model(w0, w1)
    w0, w1 = load_model()

    y_pred = feed_forward(X * 1.0, w0, w1)  # should be good since this is training data
    y_pred_num = np.argmax(y_pred, axis=1)
    accuracy = np.count_nonzero(y_pred_num == y_numbers) / len(y)
    print("Accuracy on train set", accuracy)

    y_test_pred = feed_forward(test_X * 1.0, w0, w1)
    y_test_pred_num = np.argmax(y_test_pred, axis=1)
    test_accuracy = np.count_nonzero(y_test_pred_num == test_y_numbers) / len(test_y)
    print("Accuracy on test set", test_accuracy)

    import matplotlib.pyplot as plt
    sample_test_x = test_X[:20]
    sample_test_y_num = test_y_numbers[:20]
    sample_pred_test_y_num = y_test_pred_num[:20]

    for x, y, pred in zip(sample_test_x, sample_test_y_num, sample_pred_test_y_num):
        plt.pcolor(np.reshape(x, (28, 28)), cmap='Greys')
        plt.title('Actual {}, pred {}'.format(y, pred))
        plt.show()







