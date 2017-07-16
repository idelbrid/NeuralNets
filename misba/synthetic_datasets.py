from sklearn.datasets import make_classification
from sklearn.base import clone
from itertools import product
from collections import OrderedDict

import numpy as np


def create_mil_dataset(bag_sizes, prob_inst_positive=None, negative_label=0, include_instance_labels=True,
                       random_state=123456, **make_classification_kwargs):
    """Create a MIL dataset from a SIL dataset using scikit-learn's make_classification.

    This is not an efficient function.
    This function makes an underlying strongly supervised problem using scikit-learn's make_classification(). Given
    a list of bag sizes, create a dataset with bags having such sizes. For each instance, with probability
    `prob_inst_positive` draw if from the positive examples and with probability (1 - `prob_inst_positive`) draw it
    from the negative examples dataset.

    :param bag_sizes: List of sizes for each bag in the dataset being created
    :param prob_inst_positive: Probability for a given instance to be positive. Default uses bag_sizes to make
        an approximately balanced dataset in terms of bag labels.
    :param random_state: random state passed to make_classification()
    :param make_classification_kwargs: any other key word arguments passed to make_classification
    :return: list of bags and a list of labels
    """
    np.random.seed(random_state)
    if prob_inst_positive is None:
        prob_inst_positive = 1 - np.exp(np.log(0.5)/np.mean(bag_sizes))

    X, y= make_classification(n_samples=np.asarray(bag_sizes).sum()*3, random_state=random_state, **make_classification_kwargs)
    negative = X[y==0]
    positive = X[y==1]
    neg_pos = 0
    pos_pos = 0
    bags = []
    labels = []
    for i in range(len(bag_sizes)):
        bagdata = []
        baglabels = (np.random.uniform(size=bag_sizes[i]) > (1-prob_inst_positive)).astype(int)
        for lab in baglabels:
            if lab == 0:
                bagdata.append(negative[neg_pos])
                neg_pos += 1
            else:
                bagdata.append(positive[pos_pos])
                pos_pos += 1
        bagdata = np.array(bagdata)
#         print(bagdata.shape)
#         print(baglabels.shape)
        if negative_label != 0:
            baglabels[baglabels == 0] = negative_label
        bags.append(np.hstack([bagdata, baglabels.reshape(-1, 1)]))
        labels.append(max(baglabels))
    bags = np.array(bags)
    labels = np.array(labels)

    if negative_label != 0:
        labels[labels == 0] = negative_label

    if not include_instance_labels:
        labels = remove_instance_level_labels(labels)
    return bags, labels


def create_mil_dataset_v2(bag_sizes, prob_bag_pos=0.5, prob_inst_pos=0.5, include_instance_labels=True,
                          negative_label=0, random_state=123456, **make_classification_kwargs):
    """Create a MIL dataset from a SIL dataset using scikit-learn's make_classification.

    This is not an efficient function.
    This function makes an underlying strongly supervised problem using scikit-learn's make_classification(). Given
    a list of bag sizes, create a dataset with bags having such sizes. For each instance in a positive bag, with
    probability `prob_inst_positive` draw it from the positive examples and with probability (1 - `prob_inst_positive`)
    draw it from the negative examples dataset. If none are positive, randomly choose an instance to be positive instead

    :param bag_sizes: List of sizes for each bag in the dataset being created
    :param prob_bag_pos: Probability for a bag to be positive.
    :param prob_inst_pos: Probability for a given instance of a positive bag to be positive.
    :param random_state: random state passed to make_classification()
    :param make_classification_kwargs: any other key word arguments passed to make_classification
    :return: list of bags and a list of labels
    """
    np.random.seed(random_state)
    X, y= make_classification(n_samples=np.asarray(bag_sizes).sum()*3, random_state=random_state, **make_classification_kwargs)

    negative = X[y==0]
    positive = X[y==1]
    neg_pos = 0
    pos_pos = 0
    bags = []
    labels = (np.random.uniform(size=len(bag_sizes)) < prob_bag_pos).astype(int)
    for i in range(len(bag_sizes)):
        bagdata = []
        if labels[i] == 0:
            baglabels = np.zeros([bag_sizes[i]], dtype='int')
        else:
            baglabels = (np.random.uniform(size=bag_sizes[i]) > (1-prob_inst_pos)).astype(int)
            if sum(baglabels) == 0:
                baglabels[np.random.randint(0, high=len(baglabels)-1)] = 1
        for lab in baglabels:
            if lab == 0:
                bagdata.append(negative[neg_pos])
                neg_pos += 1
            else:
                bagdata.append(positive[pos_pos])
                pos_pos += 1
        bagdata = np.array(bagdata)
#         print(bagdata.shape)
#         print(baglabels.shape)
        if negative_label != 0:
            baglabels[baglabels == 0] = negative_label
        bags.append(np.hstack([bagdata, baglabels.reshape(-1, 1)]))

    bags = np.array(bags)

    if negative_label != 0:
        labels[labels == 0] = negative_label

    if not include_instance_labels:
        labels = remove_instance_level_labels(labels)

    return bags, labels


def remove_instance_level_labels(mil_dataset):
    return np.array([np.array([inst[:-1] for inst in bag]) for bag in mil_dataset])


def extract_instance_labeled_dataset(mil_dataset):
    """Given that the first feature for each instance its true label, make the strongly supervised dataset.

    :param mil_dataset: list of bags, where each instance has its true label as the last feature
    :return: instances, instance labels

    >>> x = [[(3, 4, 5, 1), (2, 2, 3, 0)], [(2, 6, 7, 1)]]
    >>> newx, newy = extract_instance_labeled_dataset(x)
    >>> newx.tolist()
    [[3, 4, 5], [2, 2, 3], [2, 6, 7]]
    >>> newy.tolist()
    [1, 0, 1]
    """
    # sil_dataset = None
    sil_dataset = np.vstack(mil_dataset)
    # for bag in mil_dataset:
    #     if sil_dataset is None:
    #         sil_dataset = bag
    #     sil_dataset = np.vstack([sil_dataset, bag])
    return sil_dataset[:, :-1], sil_dataset[:, -1]


def extract_sil_dataset(mil_dataset, bag_labels):
    """Give each instance in each bag the label of the bag.

    :param mil_dataset: list of bags as arrays or lists
    :param bag_labels: 1-dimensional list or array of bag labels
    :return: instances, instance labels

    >>> X = [[(1,2,3), (3,4,5)], [(5,6,7)]]
    >>> Y = [0, 1]
    >>> newx, newy = extract_sil_dataset(X, Y)
    >>> newx.tolist()
    [[1, 2, 3], [3, 4, 5], [5, 6, 7]]
    >>> newy.tolist()
    [0, 0, 1]
    """
    labels = []

    instances = np.concatenate(mil_dataset)
    for bag, label in zip(mil_dataset, bag_labels):
        labels.extend([label]*len(bag))
    return instances, np.array(labels)


def run_mil_grid_search(model, param_grid, train_x, train_y, valid_x, valid_y, metric='accuracy', save_all=False,
                        negative_label=0):
    """Run a grid search over a scikit-learn-like model. Actually MIL agnostic

    :param model: scikit-learn like model, implementing scikit-learn interface for get and set params, fit, and predict
    :param param_grid: hyperparameter name to grid values dict describing the parameter grid
    :param train_x: training feature values
    :param train_y: training labels
    :param valid_x: validation feature values
    :param valid_y: validation labels
    :param metric: metric used in grid search
    :param save_all: if true, save all models and return in the output. Otherwise, model_grid returned is empty.
    :return: best_model, best_metric, model_grid, metric_grid
    """
    metric_dict = dict()
    model_dict = dict()

    best_metric = -np.inf
    best_model = None
    if not(isinstance(param_grid, list) or isinstance(param_grid, tuple) or isinstance(param_grid, np.ndarray)):
        param_grid = [param_grid]


    for param_grid_dict in param_grid:
        for param_tup in product(*param_grid_dict.values()):
            param_dict = {k: v for k, v in zip(param_grid_dict.keys(), param_tup)}

            grid_model = clone(model).set_params(**param_dict)
            grid_model.fit(train_x, train_y)
            valid_pred = grid_model.predict(valid_x)
            if metric == 'accuracy':
                cutoff = (1 + negative_label) / 2.0
                valid_pred = (valid_pred > cutoff).astype(int)
                valid_pred[valid_pred == 0] = negative_label
                grid_metric = np.mean(valid_pred == valid_y)
            else:
                raise NotImplementedError("Metrics other than accuracy are not implemented at this time")

            if grid_metric > best_metric:
                best_metric = grid_metric
                best_model = grid_model

            if save_all:
                model_dict[param_tup] = grid_model
            metric_dict[param_tup] = grid_model

    return best_model, best_metric, model_dict, metric_dict


def replace_zero(labels):
    labels = labels.copy()
    labels[labels == 0] = -1
    return labels


def replace_neg_one(labels):
    labels[labels == -1] = 0
    return labels
