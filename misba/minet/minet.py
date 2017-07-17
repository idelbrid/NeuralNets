from keras.layers import Input, Activation, Dense, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Masking, TimeDistributed, Lambda
from keras.models import Model
from keras.losses import binary_crossentropy
from keras.optimizers import SGD, Adam
from keras.preprocessing.sequence import pad_sequences
import keras

import numpy as np

from .mask_utilities import ZeroMaskedEntries, mask_aware_max, mask_aware_mean, mask_aware_mean_output_shape, logsumexp

from sklearn.base import BaseEstimator

class miNet(BaseEstimator):
    def __init__(self, hidden_layer_sizes=(30,), agg='max', activation='relu', optimizer='sgd', lr=0.01, loss='binary_crossentropy',
                 iterations=100, verbose=2):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.agg = agg
        self.activation = activation
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss
        self.model = None
        self.iterations = iterations
        self.verbose = verbose

        self.instance_model = None
        self._maxlen = None

    def _construct_model(self, input_shape):
        """Build the Keras model"""
        input_layer = Input([None]+list(input_shape[2:]))
        masked_input = Masking()(input_layer)
        last_layer = masked_input

        for size in self.hidden_layer_sizes[:]:
            last_layer = TimeDistributed(Dense(size,activation=self.activation))(last_layer)
        instance_prediction = TimeDistributed(Dense(1, activation='sigmoid'))(last_layer)

        zeroed_layer = ZeroMaskedEntries()(instance_prediction)
        if self.agg == 'max':
            agg_func = mask_aware_max
        elif self.agg == 'mean':
            agg_func = mask_aware_mean
        elif self.agg == 'logsumexp':
            agg_func = logsumexp
        else:
            raise NotImplementedError("Only max, mean, and logsumexp aggregation methods are implemented")

        output_layer = Lambda(agg_func, mask_aware_mean_output_shape)(zeroed_layer)
        self.model = Model([input_layer], [output_layer])
        self.instance_model = Model([input_layer], [instance_prediction])

        self.model.compile(self._make_optimizer(), loss=self.loss)
        # self.instance_model

    def _make_optimizer(self):
        """Utility to translate the keyword arguments into keras optimizer objects"""
        if self.optimizer == 'sgd':
            return SGD(lr=self.lr)
        elif self.optimizer == 'adam':
            return Adam(lr=self.lr)
        else:
            raise NotImplementedError("This optimizer is not implemented.")

    def predict(self, x, instancePrediction=False):
        if self.instance_model is None:
            raise ValueError("The model is not fit yet; cannot predict()")

        x = pad_sequences(x, dtype='float32', value=0, maxlen=self._maxlen)

        bag_predictions = self.model.predict(x)
        if instancePrediction:
            bag_sizes = np.sum(~(x == 0).all(axis=-1), axis=1)
            all_inst_pred = self.instance_model.predict(x)
            filtered_inst_pred = np.array([pred[:bag_len] for pred, bag_len in zip(all_inst_pred, bag_sizes)])
            return bag_predictions, np.vstack(filtered_inst_pred)
        else:
            return bag_predictions

    def fit(self, x, y):
        x = pad_sequences(x, dtype='float32', value=0)
        self._maxlen = len(x[0])

        self._construct_model(x.shape)
        self.model.fit(x, y, verbose=self.verbose, epochs=self.iterations)
        return self

    def get_parameters(self):
        return {
            'inst_hidden_layer_sizes': self.hidden_layer_sizes,
            'agg': self.agg,
            'activation': self.optimizer,
            'lr': self.lr,
            'loss': self.loss,
            'iterations': self.iterations,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


class MINet(BaseEstimator):
    def __init__(self, inst_hidden_layer_sizes=(30,), bag_hidden_layer_sizes=(30,), agg='max', activation='relu',
                 optimizer='sgd', lr=0.01, loss='binary_crossentropy', iterations=100, verbose=2):
        self.inst_hidden_layer_sizes = inst_hidden_layer_sizes
        self.bag_hidden_layer_sizes = bag_hidden_layer_sizes
        self.agg = agg
        self.activation = activation
        self.optimizer = optimizer
        self.lr = lr
        self.loss = loss
        self.model = None
        self.iterations = iterations
        self.verbose = verbose

        # self.instance_model = None
        self._maxlen = None

    def _construct_model(self, input_shape):
        """Build the Keras model"""
        input_layer = Input([None]+list(input_shape[2:]))
        masked_input = Masking()(input_layer)
        last_layer = masked_input

        for size in self.inst_hidden_layer_sizes[:]:
            last_layer = TimeDistributed(Dense(size,activation=self.activation))(last_layer)

        zeroed_layer = ZeroMaskedEntries()(last_layer)
        if self.agg == 'max':
            agg_func = mask_aware_max
        elif self.agg == 'mean':
            agg_func = mask_aware_mean
        elif self.agg == 'logsumexp':
            agg_func = logsumexp
        else:
            raise NotImplementedError("Only max, mean, and logsumexp aggregation methods are implemented")

        encoded_layer = Lambda(agg_func, mask_aware_mean_output_shape)(zeroed_layer)
        last_layer = encoded_layer
        for size in self.bag_hidden_layer_sizes:
            last_layer = Dense(size, activation=self.activation)(last_layer)
        output_layer = Dense(1, activation='sigmoid')(last_layer)

        self.model = Model([input_layer], [output_layer])
        self.model.compile(self._make_optimizer(), loss=self.loss)

    def _make_optimizer(self):
        """Utility to translate the keyword arguments into keras optimizer objects"""
        if self.optimizer == 'sgd':
            return SGD(lr=self.lr)
        elif self.optimizer == 'adam':
            return Adam(lr=self.lr)
        else:
            raise NotImplementedError("This optimizer is not implemented.")

    def predict(self, x, instancePrediction=False):
        if self.model is None:
            raise ValueError("The model is not fit yet; cannot predict()")
        if instancePrediction:
            raise ValueError("The MINet model does not have an instance-level prediction")

        x = pad_sequences(x, dtype='float32', value=0, maxlen=self._maxlen)

        bag_predictions = self.model.predict(x)
        return bag_predictions

    def fit(self, x, y):
        x = pad_sequences(x, dtype='float32', value=0)
        self._maxlen = len(x[0])

        self._construct_model(x.shape)
        self.model.fit(x, y, verbose=self.verbose, epochs=self.iterations)
        return self

    def get_parameters(self):
        return {
            'inst_hidden_layer_sizes': self.inst_hidden_layer_sizes,
            'bag_hidden_layer_sizes': self.bag_hidden_layer_sizes,
            'agg': self.agg,
            'activation': self.optimizer,
            'lr': self.lr,
            'loss': self.loss,
            'iterations': self.iterations,
            'verbose': self.verbose,
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
