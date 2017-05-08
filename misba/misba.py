import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class MISBA:
    def __init__(self, mi_structure):
        """
        mi_structure should be of the form [(N_1, M_1), ..., (N_K, M_K)] where K is the number of kinds of instances,
          N_i is the number of instances for the ith kind of instance, and M_i is the number of features for said 
          kind of instance.
           
        """
        self.mi_structure = mi_structure

    def pretrain(self, X):
        raise NotImplementedError()

    def fine_tune(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        raise NotImplementedError()

    def encode(self, X):
        raise NotImplementedError()

    def reconstruct(self, X):
        raise NotImplementedError()

    def get_reconstruction_errors(self, X):
        raise NotImplementedError()


if __name__ == '__main__':
    pass
