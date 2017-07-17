from synthetic_datasets import *
from scipy.io import loadmat, savemat
import pandas as pd
import sys
svm_grid = [
    {
        'C': [0.001, 0.01, 0.1, 1.],
        'kernel': ['linear', 'quadratic'],
        'scale_C': [True, False],
    },
    {
        'C': [0.001, 0.01, 0.1, 1.],
        'kernel': ['rbf'],
        'gamma': [0.001, 0.01, 0.1, 1.],
        'scale_C': [True, False],
    }]

sil_grid = [
    {
        'C': [0.001, 0.01, 0.1, 1.],
        'kernel': ['linear', 'quadratic'],
        'scale_C': [True, False],
    },
    {
        'C': [0.001, 0.01, 0.1, 1.],
        'kernel': ['rbf'],
        'gamma': [0.001, 0.01, 0.1, 1.],
        'scale_C': [True, False],
    }]


miNet_grid = {
    'activation': ['relu', 'sigmoid'],
    'agg': ['max', 'mean', 'logsumexp'],
    'hidden_layer_sizes': [[], [30], [100], [100, 100]],
    'lr': [0.01, 0.001],
    'iterations': [100, 500]
}

MINet_grid = {
    'activation': ['relu', 'sigmoid'],
    'agg': ['max', 'mean', 'logsumexp'],
    'inst_hidden_layer_sizes': [[], [30], [100], [100, 100]],
    'bag_hidden_layer_sizes': [[], [30], [100], [100, 100]],
    'lr': [0.01, 0.001],
    'iterations': [100, 500]
}

test_grid = {
    5: np.arange(1,6,1)/5,
    20: np.arange(1, 21, 1) / 20,
    100: np.arange(1, 101, 2) / 100,
    500: np.arange(1, 501, 5) / 500
}
class DatasetWrapper:
    def __init__(self, xlabeled, bag_labels):
        self.xlabeled = xlabeled
        self.y = bag_labels
        self.x = remove_instance_level_labels(xlabeled)

        self.x_inst, self.y_inst = extract_instance_labeled_dataset(xlabeled)


def make_dataset(bag_size, proportion_positive, negative_label=-1):
    xlabeled, y = create_mil_dataset_v2([bag_size]*1200, prob_inst_pos=proportion_positive, negative_label=negative_label)

    train = DatasetWrapper(xlabeled[:200], y[:200])
    valid = DatasetWrapper(xlabeled[200:700], y[200:700])
    test = DatasetWrapper(xlabeled[700:], y[700:])

    return train, valid, test


def accuracy(model, test_x, test_y, negative_label=-1, instance=False):
    if instance:
        _, test_pred = model.pred(test_x, instance)
    else:
        test_pred = model.pred(test_x)

    if negative_label == -1:
        test_pred = np.sign(test_pred)
    else:
        test_pred = (test_pred > 0.5).astype(int)
    return np.mean(test_pred == test_y)


def run_model(base_model, test_grid, param_grid, name, negative_label=-1, instancePrediction=False):
    results_df = pd.DataFrame()
    valid_results_df = pd.DataFrame()
    results_inst_df = pd.DataFrame()
    valid_results_inst_df = pd.DataFrame()

    for bag_size, proportion_range in test_grid.items():
        print("Starting bag size {}".format(bag_size))
        for proportion in proportion_range:
            print("Starting proportion {}".format(proportion))
            train, valid, test = make_dataset(bag_size, proportion, negative_label)
            best_model, best_score, all_models, all_scores = run_mil_grid_search(base_model, param_grid, train.x,
                                                                                 train.y, valid.x, valid.y,
                                                                                 negative_label=negative_label)
            test_accuracy = accuracy(best_model, test.x, test.y, negative_label)
            results_df.set_value(bag_size, proportion, test_accuracy)
            valid_results_df.set_value(bag_size, proportion, best_score)

            if instancePrediction:
                test_inst_accuracy = accuracy(best_model, test.x, test.y_inst, negative_label=negative_label,
                                              instance=True)
                valid_inst_accuracy = accuracy(best_model, valid.x, valid.y_inst, negative_label=negative_label,
                                              instance=True)
                results_inst_df.set_value(bag_size, proportion, test_inst_accuracy)
                valid_results_inst_df.set_value(bag_size, proportion, valid_inst_accuracy)

    results_df.to_csv('RESULTS_bag_size_proportion_{}.csv'.format(name))
    valid_results_df.to_csv('RESULTS_bag_size_proportion_{}_validation.csv'.format(name))
    if instancePrediction:
        results_inst_df.to_csv('RESULTS_bag_size_proportion_{}_instance.csv'.format(name))
        valid_results_inst_df.to_csv('RESULTS_bag_size_proportion_{}_instance_validation.csv'.format(name))


if __name__ == '__main__':
    model_name = sys.argv[1]
    if model_name == 'miSVM':
        from misvm import miSVM
        model = miSVM(verbose=0, max_iters=500, restarts=3)
        run_model(model, test_grid, svm_grid, model_name, negative_label=-1, instancePrediction=True)

    elif model_name == 'MISVM':
        from misvm import MISVM
        model = MISVM(verbose=0, max_iters=500, restarts=3)
        run_model(model, test_grid, svm_grid, model_name, negative_label=-1, instancePrediction=False)


    elif model_name == 'SIL':
        from misvm import SIL
        model = SIL(verbose=0, max_iters=500, restarts=3)
        run_model(model, test_grid, sil_grid, model_name, negative_label=-1, instancePrediction=True)


    elif model_name == 'miNet':
        from minet import miNet
        model = miNet(verbose=0)
        run_model(model, test_grid, miNet_grid, model_name, negative_label=0, instancePrediction=True)


    elif model_name == 'MINet':
        from minet import MINet
        model = MINet(verbose=0)
        run_model(model, test_grid, MINet_grid, model_name, negative_label=0, instancePrediction=False)


    else:
        raise ValueError("Unrecognized model name")