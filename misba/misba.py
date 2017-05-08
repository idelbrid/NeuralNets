import Tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class MISBA:
	def __init__(self, mi_structure):
		self.mi_structure = mi_structure
	
	def pretrain(X):
		raise NotImplementedError()
	
	def fine_tune(X, y):
		raise NotImplementedError()

	def predict(X):
		raise NotImplementedError()

	def encode(X):
		raise NotImplementedError()

	def reconstruct(X):
		raise NotImplementedError()

	def get_reconstruction_errors(X):
		raise NotImplementedError()


