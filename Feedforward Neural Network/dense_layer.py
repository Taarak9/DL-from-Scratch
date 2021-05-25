import numpy as np
import random
from activation_functions import activation_function

# Dense Layer
class Dense_Layer:

  # Initialize weights and biases
  def __init__(self, n_inputs, n_nodes):
    # He initialization and transposed 
    self.weights = ​np.random.randn(n_inputs, n_neurons) * np.sqrt(2 / n_nodes)
    # Zero initialization
    self.biases = np.zeros((​1, n_nodes))

  # forward pass 
  # activation type: type of activation function
  def forward(self, x, activation_type):
    for b, w in zip(self.biases, self.weights): 
      x = activation_function(activation_type, np.dot(w, x) + b)
    return x