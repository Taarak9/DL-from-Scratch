import numpy as np
import random
from matplotlib import pyplot as plt

# Dense Layer
class Dense_Layer:

  # Initialize weights and biases
  def __init__(self, n_inputs, n_nodes, activation_type):
    self.n_nodes = n_nodes
    self.activation_type = activation_type

# Feedforward Neural network
class FNN():

  def __init__(self, n_inputs, loss_fn):
    self.NN = list()
    #sizes contains the list of number of neurons in the respective layers of the network.
    self.sizes = [n_inputs]
    self.n_layers = 0
    self.n_inputs = n_inputs
    self.weights = list()
    self.biases = list()
    self.prev_update_w = list()
    self.prev_update_b = list()
    # activation_types contains the list of activation functions used in the NN
    self.activation_types = list()
    self.epoch_list = list()
    self.loss_fn = loss_fn
    self.accuracy = list()

  def init_params(self, sizes, epochs):
    self.sizes = sizes
    self.n_layers = len(sizes)
    # he initialization
    self.weights = [np.random.randn(y, x) * np.sqrt(2 / x) for x, y in zip(sizes[:-1], sizes[1:])]
    # zero initialization
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    self.epoch_list = np.arange(0, epochs)

  def get_params(self):
    return [self.weights, self.biases]

  # add layer to fnn
  def add_layer(self, n_nodes, activation_type):
    if self.n_layers == 0:
        # previous layer is input layer
        layer = Dense_Layer(self.n_inputs, n_nodes, activation_type)
    else:
        # previous layer is hidden layer
        layer = Dense_Layer(self.sizes[-1], n_nodes, activation_type)

    self.NN.append(layer)
    self.activation_types.append(activation_type)
    self.sizes.append(n_nodes)
    self.n_layers += 1

  # a is the input to the nn
  def feedforward(self, a):
    l = 0 # layer count
    for b, w in zip(self.biases, self.weights): 
      a = activation_function(self.activation_types[l], np.dot(w, a) + b)
      l += 1
    return a

  # number of matches between output from NN and test data
  def evaluate(self, test_data, type):      
    if type == "classification":
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    elif type == "regression":
        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
    else:
        pass
    return sum(int(x == y) for (x, y) in test_results)

  # Backpropagation for one example
  def backprop(self, x, y, weights=None, biases=None):
    if weights: self.weights = weights
    if biases: self.biases = biases

    gradient_w = [np.zeros(w.shape) for w in self.weights]
    gradient_b = [np.zeros(b.shape) for b in self.biases]

    # feedforward
    activation = x
    # list to store all the activations, layer by layer
    activations = [x] 
    # list to store all the z vectors, layer by layer
    zs = [] 
    # c: layer counter
    c = 0
    for w, b in zip(self.weights, self.biases):
        z = np.dot(w, activation) + b
        zs.append(z)
        activation = activation_function(self.activation_types[c], z)
        activations.append(activation)
        c += 1

    loss_grad = loss_function(self.loss_fn, y, activations[-1], True)
    # delta: errors of the output layer
    delta = loss_grad * activation_function(self.activation_types[-1], zs[-1], True)
    
    gradient_w[-1] = np.dot(delta, activations[-2].transpose())
    gradient_b[-1] = delta
    # backpropagate the error
    for l in range(2, self.n_layers):
        z = zs[-l]
        d = activation_function(self.activation_types[-l], z, True)
        # Here delta is errors of the layer n_layers - l
        delta = np.dot(self.weights[-l + 1].transpose(), delta) * d
        gradient_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        gradient_b[-l] = delta

    return (gradient_w, gradient_b)


  def get_batch_size(self, training_data, mode, mini_batch_size):
    if mode == "online":
        return 1
    elif mode == "mini_batch":
        return mini_batch_size
    elif mode == "batch":
        return len(training_data)

  def update_GD(self, mini_batch, eta):
    gradient_b = [np.zeros(b.shape) for b in self.biases]
    gradient_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_gradient_b, delta_gradient_w = self.backprop(x, y)
        gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
        gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

    self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, gradient_w)]
    self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, gradient_b)]

  def update_MGD(self, mini_batch, gamma, eta):
    gradient_b = [np.zeros(b.shape) for b in self.biases]
    gradient_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_gradient_b, delta_gradient_w = self.backprop(x, y)
        gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]
        gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]

    update_w = gamma * self.prev_update_w + eta * gradient_w
    self.weights = [w - uw for w, uw in zip(self.weights, update_w)]
    update_b = gamma * self.prev_update_b + eta * gradient_b
    self.biases = [b - ub for b, ub in zip(self.biases, update_b)]

    self.prev_update_w = update_w
    self.prev_update_b = update_b

  def update_NAG(self, mini_batch, eta, gamma):
    gradient_w = [np.zeros(w.shape) for w in self.weights]
    gradient_b = [np.zeros(b.shape) for b in self.biases]

    # w look_ahead partial update
    update_w = gamma * self.prev_update_w
    update_b = gamma * self.prev_update_b

    for x, y in mini_batch:
        delta_gradient_w, delta_gradient_b = FNN.backprop(x, y, self.weights - update_w, self.biases - update_b)
        gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]
        gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]       

    # full update
    update_w = gamma * self.prev_update_w + eta * gradient_w
    self.weights = [w - uw for w, uw in zip(self.weights, update_w)]
    update_b = gamma * self.prev_update_b + eta * gradient_b
    self.biases = [b - ub for b, ub in zip(self.biases, update_b)]
    
    self.prev_update_w = update_w
    self.prev_update_b = update_b

  '''
  epochs: max epochs
  eta - learning rate
  optimizer: GD ( Gradient Descent), MGD ( Momentum based GD ), NAG ( Nesterov accelerated GD )
  mode: online ( Stochastic GD ), mini-batch ( Mini-batch GD ), batch ( Batch GD)
  shuffle: True/False
  gamma - momentum value
  task: classification/regression
  '''
  def Optimizer(self, training_data, epochs, mini_batch_size, eta, gamma=None, optimizer="GD", mode="batch", shuffle=True, test_data=None, task=None):
    n = len(training_data)
    batch_size = self.get_batch_size(training_data, mode, mini_batch_size)

    if optimizer == "MGD":
        self.prev_update_w = [np.zeros(w.shape) for w in self.weights]
        self.prev_update_b = [np.zeros(b.shape) for b in self.biases]
    
    for e in range(epochs):
        if shuffle:
            random.shuffle(training_data)
        
        mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
        
        for mini_batch in mini_batches:
            if optimizer == "GD":
                self.update_GD(mini_batch, eta)
            elif optimizer == "MGD":
                self.update_MGD(mini_batch, eta, gamma)
            elif optimizer == "NAG":
                self.update_NAG(mini_batch, eta, gamma)
        
        if test_data:
            FNN.tracking(e, epochs, test_data, task)

  def tracking(self, epoch, epochs, test_data=None, task=None):
    if test_data:
        print("Epoch: ", epoch, "Accuracy: ", self.evaluate(test_data, task) / len(test_data) * 100)
        self.accuracy.append(self.evaluate(test_data, task) / len(test_data) * 100)
        if epoch == epochs - 1:
              print("Max accuracy achieved: ", np.around(np.max(self.accuracy), decimals=2), 
                    "at epoch ", self.epoch_list[np.argmax(self.accuracy)])
    else:
        print("Epoch {0} complete".format(epoch))

  def logging(self, test_data=None):
    if test_data:
        error = [(100 - a) for a in self.accuracy ]

        plt.plot(self.epoch_list, error)
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()
    else:
        pass

  def compile(self, training_data, epochs, mini_batch_size, eta, gamma=None, optimizer="GD", mode="batch", shuffle=True, test_data=None, task=None):
    self.init_params(self.sizes, epochs)

    self.Optimizer(training_data, epochs, mini_batch_size, eta, gamma, optimizer, mode, shuffle, test_data, task)
      
    if test_data:
        self.logging(test_data)

# y and y_hat are list of numpy arrays
def loss_function(name, y, y_hat, derivative=False):
  # y - target, y_hat - output
  # Mean Squared Error
  if name == "mse":
      if derivative:
          return (y_hat - y)
      else:
          return np.mean((y - y_hat)**2)
  # y - target prob distro, y_hat - output prob distro
  # Cross Entropy
  elif name == "ce":
      if derivative:
          # if activation fn is sigmoid/softmax
          return (y_hat - y)    
      else:
          return -np.sum(y * np.log(y_hat))

def activation_function(name, input, derivative=False):
  if name == "identity":
      if derivative:
          return np.ones_like(x)
      else:
          return x
  elif name == "sigmoid":
      if derivative:
          out = activation_function(name, input)
          return out * ( 1 - out )
      else:
          return 1 / (1 + np.exp(-input))
  elif name == "softmax":
      if derivative:
          out = activation_function(name, input)
          return out * (1 - out)
      else:
          e_x = np.exp(input - np.max(input))
          return e_x / np.sum(e_x, axis=1, keepdims=True)
  elif name == "tanh":
      if derivative:
          out = activation_function(name, input)
          return 1 - np.square(out)
      else:
          return 2 / (1 + np.exp(-2*input)) - 1
  elif name == "relu":
      if derivative:
          return (input > 0) * 1
      else:
          return np.maximum(0, input)
