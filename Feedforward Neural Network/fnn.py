import numpy as np
import random
from activation_functions import activation_function
from loss_functions import loss_function

# Feedforward Neural Network
class FNN(object):
  
  def __init__(self, sizes):
    #sizes contains the list of number of neurons in the respective layers of the network.
    self.nlayers = len(sizes)
    self.sizes = sizes
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

  def feedforward(self, a):
    for b, w in zip(self.biases, self.weights): 
      a = sigmoid(np.dot(w, a) + b)
    return a

  def evaluate(self, test_data):      
    test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

  # Backpropagation
  def backprop(self, x, y):
      nabla_b = [np.zeros(b.shape) for b in self.biases]
      nabla_w = [np.zeros(w.shape) for w in self.weights]

      # feedforward
      activation = x
      activations = [x] # list to store all the activations, layer by layer
      zs = [] # list to store all the z vectors, layer by layer
      for b, w in zip(self.biases, self.weights):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = activation_function("sigmoid", z)
        activations.append(activation)

      # backward pass
      delta = loss_function("ce", y, activations[-1], True) * \
            activation_function("sigmoid", zs[-1], True)
      nabla_b[-1] = delta
      nabla_w[-1] = np.dot(delta, activations[-2].transpose())
      for l in range(2, self.nlayers):
        z = zs[-l]
        sp = activation_function("sigmoid", z, True)
        delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
      return (nabla_b, nabla_w)

  def update_mini_batch(self, mini_batch, eta):
      nabla_b = [np.zeros(b.shape) for b in self.biases]
      nabla_w = [np.zeros(w.shape) for w in self.weights]
      for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

      self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
      self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

  # Stochastic Gradient decsent
  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in range(epochs):
      random.shuffle(training_data)
      mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      if test_data:
        print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
      else:
        print("Epoch {0} complete".format(j))
