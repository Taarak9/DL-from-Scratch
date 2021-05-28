# -*- coding: utf-8 -*-
"""optimizers

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PCYUoc6Upe4Pum_WLnuzEo0Ml66WvOys
"""

import numpy as np
from generalized_fnn import FNN

class Optimizers:

  def __init__(self):
    self.weights, self.biases = FNN.get_params()

  def get_params():
    return [self.weights, self.biases]

  def update_mini_batch(self, mini_batch, eta):
      gradient_b = [np.zeros(b.shape) for b in self.biases]
      gradient_w = [np.zeros(w.shape) for w in self.weights]
      for x, y in mini_batch:
          delta_gradient_b, delta_gradient_w = FNN.backprop(x, y)
          gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
          gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

      self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, gradient_w)]
      self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, gradient_b)]

  # Stochastic Gradient decsent
  # In each epoch, it starts by randomly shuffling the training data, and then partitions it into mini-batches.
  # Then for each mini_batch we apply a single step of gradient descent, which updates the network weights and biases.
  '''
  eta - learning rate
  task - Classification / Regression
  '''
  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None, task=None):
    self.init_params(self.sizes, epochs)
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            self.update_mini_batch(mini_batch, eta)