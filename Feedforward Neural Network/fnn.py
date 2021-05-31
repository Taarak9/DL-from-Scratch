import numpy as np
import random
from matplotlib import pyplot as plt

class Dense_Layer:
  """Dense Layer Class. """
  def __init__(self, n_nodes, activation_type):
    """
    Creates the Dense layer.

    Parameters
    ----------
    n_nodes: int
        Number of nodes in the layer.
    activation_type: str
        Activation function used in the layer.
        Options:
            identity,
            sigmoid,
            softmax,
            tanh,
            relu
    """

    self.n_nodes = n_nodes
    self.activation_type = activation_type

class FNN():
  """
  Feedforward Neural network Class.

  Attributes
  ----------
  NN: list
      List of all the layers in the NN.

  sizes: list
      List of number of nodes in the respective layers of the NN.

  n_layers: int
      Total number of layers in the NN.

  n_inputs: int
      Number of nodes in the input layer.

  weights: list of numpy arrays
      Each array has the weights in a particular layer.

  biases: list of numpy arrays
      Each array has the biases in a particular layer.

  prev_update_w: list
      List of all previous updates of weights in NN.

  prev_update_b: list
      List of all previous updates of weights in NN.

  activation_types: list
      List of all the activation functions used
      in the respective layers of NN.

  loss_fn: str
      Type of loss function used in the NN.
      Options:
          mse ( Mean squared error )
          ce ( Cross entropy )

  epoch_list: list
      List of numbers from 0 to max epochs.

  accuracy: list
      List to store the accuracy at each epoch.

  Methods
  -------
  init_params(sizes, epochs):
      Initializes parameters in the NN.

  get_params():
      Return weights and biases of the NN.

  add_layer(n_nodes, activation_type):
      Adds a layer to the NN.

  feedforward(a):
      Return the output of NN if input is a. 

  evaluate(test_data, task):      
      Return the number of test inputs for which the
      NN outputs the correct result.

  backprop(x, y, weights=None, biases=None):
      Returns the gradients of weights and biases
      for a given example.

  get_batch_size(training_data, mode, mini_batch_size):
      Returns the batch size given mode.

  update_GD(mini_batch, eta):
      Updates weights and biases after 
      applying Gradient Descent (GD)
      on the mini batch.

  update_MGD(mini_batch, gamma, eta):
      Updates weights and biases after 
      applying Momentum based Gradient Descent (MGD)
      on the mini batch.

  update_NAG(mini_batch, eta, gamma):
      Updates weights and biases after 
      applying Nesterov accerelated Gradient Descent (NAG)
      on the mini batch.

  Optimizer(training_data, epochs, mini_batch_size, eta, gamma=None, optimizer="GD", mode="batch", shuffle=True, test_data=None, task=None):
      Runs the optimizer on the training data
      for given number of epochs.

  tracking(epoch, epochs, test_data=None, task=None):
      If test data is given it tracks the 
      accuracy after each epoch.
      If test data isn't given it tracks
      the number of epochs completed.

  logging(test_data=None):
      Given test data it plots 
      Epoch vs Error graph.

  compile(training_data, epochs, mini_batch_size, eta, gamma=None, optimizer="GD", mode="batch", shuffle=True, test_data=None, task=None):
      Compiles the NN i.e
      Initializes the parameters
      and runs the optimizer.
  """

  def __init__(self, n_inputs, loss_fn):
    """
    Creates the Feedforward Neural Network

    Parameters
    ----------
    n_inputs: int
        Number of nodes in the input layer.

    loss_fn: str
      Type of loss function used in the NN.
      Options:
          mse ( Mean squared error )
          ce ( Cross entropy )
    """

    self.NN = list()
    self.sizes = [n_inputs]
    self.n_layers = 0
    self.n_inputs = n_inputs
    self.weights = list()
    self.biases = list()
    self.prev_update_w = list()
    self.prev_update_b = list()
    self.activation_types = list()
    self.loss_fn = loss_fn
    self.epoch_list = list()
    self.accuracy = list()

  def init_params(self, sizes, epochs):
    """
    Initializes parameters in the NN.

    The weights and biases for the NN
    are initialized randomly, using a Gaussian
    distribution with mean 0, and variance 1.

    For weights he initialization is also used.

    Parameters
    ----------
    sizes: list
        List of number of nodes in the respective layers of the NN.

    epochs: int
        Number of maximum epochs 

    Returns
    ------
    None
    """

    #self.sizes = sizes
    self.n_layers = len(sizes)
    # he initialization
    self.weights = [np.random.randn(y, x) * np.sqrt(2 / x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    
    self.epoch_list = np.arange(0, epochs)

  def get_params(self):
    """
    Return weights and biases of the NN.
    
    Returns
    -------
    List of weights and biases
    """
    return [self.weights, self.biases]


  def add_layer(self, n_nodes, activation_type):
    """
    Adds a layer to the NN.
   
    Parameters
    ----------
    n_nodes: int
        Number of nodes in the layer.

    activation_type: str
        Activation function used in the layer.
        Options:
            identity
            sigmoid
            softmax
            tanh
            relu

    Returns
    -------
    None
    """

    layer = Dense_Layer(n_nodes, activation_type)

    self.NN.append(layer)
    self.activation_types.append(activation_type)
    self.sizes.append(n_nodes)
    self.n_layers += 1

  def feedforward(self, a):
    """
    Return the output of NN if input is a. 

    Parameter
    ---------
    a: array
        Inputs to the NN.

    Returns
    -------
    a: array
        Output of NN
    """

    l = 0 # layer count
    for b, w in zip(self.biases, self.weights): 
      a = activation_function(self.activation_types[l], np.dot(w, a) + b)
      l += 1
    return a

  def evaluate(self, test_data, task): 
    """
    Return the number of test inputs for which the NN outputs the correct result.
   
    Parameters
    ----------
    test_data: list
        List of tuples (x, y)

    type: str
        Type of task.
        Options:
            classification
            regression

    Returns
    -------
    Returns int 
    """

    if task == "classification":
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    elif task == "regression":
        test_results = [(self.feedforward(x), y) for (x, y) in test_data]
    else:
        return -1
    return sum(int(x == y) for (x, y) in test_results)

  def backprop(self, x, y, weights=None, biases=None):
    """
    Returns the gradients of weights and biases for a given example.

    Backpropagates the error.

    Parameters
    ----------
    x: tuple
        Input
    
    y: tuple
        Output

    weights: list
        Default: None

    biases: list
        Default: None

    Returns
    -------
    (gradient_w, gradient_b): tuple of lists of numpy arrays
    """

    if weights: self.weights = weights
    if biases: self.biases = biases

    gradient_w = [np.zeros(w.shape) for w in self.weights]
    gradient_b = [np.zeros(b.shape) for b in self.biases]

    activation = x
    # list to store all the activations, layer by layer
    activations = [x] 
    # list to store all the z vectors, layer by layer
    zs = [] 
    # c: layer counter
    c = 0
    # feedforward
    for b, w in zip(self.biases, self.weights):
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
        gradient_b[-l] = delta
        gradient_w[-l] = np.dot(delta, activations[-l - 1].transpose())

    return (gradient_w, gradient_b)


  def get_batch_size(self, training_data, mode, mini_batch_size):
    """
    Returns the batch size given mode.

    Parameters
    ----------
    training_data: list
        List of tuples (x, y)

    mode: str
        Options:
            online
            mini_batch
            batch
    
    mini_batch_size: int
        Size of the mini_batch

    Returns
    -------
    Returns int ( batch size )
    """

    if mode == "online":
        return 1
    elif mode == "mini_batch":
        return mini_batch_size
    elif mode == "batch":
        return len(training_data)

  def update_GD(self, mini_batch, eta):
    """
    Updates parameters using GD.

    Updates weights and biases after 
    applying Gradient Descent (GD)
    on the mini batch.

    Parameters
    ----------
    mini_batch: list
        List of tuples (x, y)

    eta: float
        Learning rate

    Returns
    -------
    None
    """

    gradient_b = [np.zeros(b.shape) for b in self.biases]
    gradient_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_gradient_w, delta_gradient_b = self.backprop(x, y)
        gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
        gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]

    self.weights = [w - (eta / len(mini_batch)) * gw 
            for w, gw in zip(self.weights, gradient_w)]
    self.biases = [b - (eta / len(mini_batch)) * gb 
            for b, gb in zip(self.biases, gradient_b)]
    

  def update_MGD(self, mini_batch, eta, gamma):
    """
    Updates parameters using MGD.

    Updates weights and biases after applying 
    Momentum based Gradient Descent (GD)
    on the mini batch.

    Parameters
    ----------
    mini_batch: list
        List of tuples (x, y)

    eta: float
        Learning rate

    gamma: float
        Momentum value

    Returns
    -------
    None
    """

    gradient_b = [np.zeros(b.shape) for b in self.biases]
    gradient_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_gradient_w, delta_gradient_b = self.backprop(x, y)
        gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]
        gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]

    update_w = gamma * self.prev_update_w + eta * gradient_w
    self.weights = [w - uw for w, uw in zip(self.weights, update_w)]
    update_b = gamma * self.prev_update_b + eta * gradient_b
    self.biases = [b - ub for b, ub in zip(self.biases, update_b)]

    self.prev_update_w = update_w
    self.prev_update_b = update_b

  def update_NAG(self, mini_batch, eta, gamma):
    """
    Updates parameters using NAG.

    Updates weights and biases after applying 
    Nesterov accerelated Gradient Descent (GD)
    on the mini batch.

    Parameters
    ----------
    mini_batch: list
        List of tuples (x, y)

    eta: float
        Learning rate

    gamma: float
        Momentum value

    Returns
    -------
    None
    """

    gradient_w = [np.zeros(w.shape) for w in self.weights]
    gradient_b = [np.zeros(b.shape) for b in self.biases]

    # w look_ahead partial update
    update_w = gamma * self.prev_update_w
    update_b = gamma * self.prev_update_b

    for x, y in mini_batch:
        delta_gradient_w, delta_gradient_b = self.backprop(x, y, self.weights - update_w, self.biases - update_b)
        gradient_w = [gw + dgw for gw, dgw in zip(gradient_w, delta_gradient_w)]
        gradient_b = [gb + dgb for gb, dgb in zip(gradient_b, delta_gradient_b)]       

    # full update
    update_w = gamma * self.prev_update_w + eta * gradient_w
    self.weights = [w - uw for w, uw in zip(self.weights, update_w)]
    update_b = gamma * self.prev_update_b + eta * gradient_b
    self.biases = [b - ub for b, ub in zip(self.biases, update_b)]
    
    self.prev_update_w = update_w
    self.prev_update_b = update_b


  def Optimizer(self, training_data, epochs, mini_batch_size=None, eta=1, gamma=None, optimizer="GD", mode="batch", shuffle=True, test_data=None, task=None):
    """
    Runs the optimizer on the training data for given number of epochs.
    
    Parameters
    ----------
    training_data: list
        List of tuples (x, y)

    epochs: int
        Maximum number of epochs.

    mini_batch_size: int
        Size of the mini_batch

    eta: float
        Learning rate.

    gamma: float
        Momentum value
        Default: None   

    optimizer: str
        Type of optimizer
        Options:
            GD ( Gradient Descent)
            MGD ( Momentum based GD )
            NAG ( Nesterov accelerated GD )
        Default: GD

    mode: str
        Mode of Learning
        Options:
            online ( Stochastic GD )
            mini-batch ( Mini-batch GD )
            batch ( Batch GD)
        Default: batch

    shuffle: bool
        Random shuffle the training data.
        Default: True

    test_data: list
        List of tuples (x, y)

    type: str
        Type of task.
        Options:
            classification
            regression

    Returns
    -------
    None
    """

    n = len(training_data)
    batch_size = self.get_batch_size(training_data, mode, mini_batch_size)

    if optimizer == "MGD":
        self.prev_update_w = [np.zeros(w.shape) for w in self.weights]
        self.prev_update_b = [np.zeros(b.shape) for b in self.biases]
    
    print("---------------Status---------------")
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
            #FNN.tracking(e, epochs, test_data, task)
            print("Epoch: ", e, "Accuracy: ", self.evaluate(test_data, task) / len(test_data) * 100)
            self.accuracy.append(self.evaluate(test_data, task) / len(test_data) * 100)
            if e == epochs - 1:
                print("Max accuracy achieved: ", np.around(np.max(self.accuracy), decimals=2), 
                        "at epoch ", self.epoch_list[np.argmax(self.accuracy)])
        else:
            print("Epoch {0} complete".format(e))

  def tracking(self, epoch, epochs, test_data=None, task=None):
    """
    Tracks the status.

    If test data is given it tracks the accuracy after each epoch.
    If test data isn't given it tracks the number of epochs completed.

    Parameters
    ----------
    epoch: int
        Current epoch

    epochs: int
        Maximum number of epochs. 

    test_data: list
        List of tuples (x, y)

    type: str
        Type of task.
        Options:
            classification
            regression

    Returns
    -------
    None
    """

    if test_data:
        print("Epoch: ", epoch, "Accuracy: ", self.evaluate(test_data, task) / len(test_data) * 100)
        self.accuracy.append(self.evaluate(test_data, task) / len(test_data) * 100)
        if epoch == epochs - 1:
              print("Max accuracy achieved: ", np.around(np.max(self.accuracy), decimals=2), 
                    "at epoch ", self.epoch_list[np.argmax(self.accuracy)])
    else:
        print("Epoch {0} complete".format(epoch))

  def logging(self, test_data=None):
    """
    Given test data it plots Epoch vs Error graph.

    Parameter
    ---------
    test_data: list
        List of tuples (x, y)

    Returns
    -------
    None
    """

    if test_data:
        error = [(100 - a) for a in self.accuracy ]

        plt.plot(self.epoch_list, error)
        plt.title("Epoch vs Error")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()
    else:
        pass

  def compile(self, training_data, test_data=None):
    """
    Compiles the NN.
      
    Initializes the parameters, runs the optimizer
    and plots epoch vs error graph given test data.
    
    Parameters
    ----------
    training_data: list
        List of tuples (x, y)

    test_data: list
        List of tuples (x, y)

    Returns
    -------
    None
    """

    epochs = int(input("Number of epochs: "))

    print("Optimizer types available:")
    print("GD (Gradient Desecent)")
    print("MGD (Momentum based Gradient Desecent)")
    print("NAG (Nesterov accerelated Gradient Desecent)")
    optimizer = input("Optimizer: ")
    if (optimizer == "MGD" or optimizer == "NAG"):
        gamma = float(input("gamma (momentum): "))
    else:
        gamma = None
    
    eta = float(input("Learning rate: "))
    
    mode = input("learning mode (online/mini_batch/batch): ")
    if mode == "mini_batch":
        mini_batch_size = int(input("Mini-batch size: "))
    else:
        mini_batch_size = None
    
    shuffle = bool(input("Random shuffle training data (True/False): "))

    task = input("task (classification/regression): ")
    
    self.init_params(self.sizes, epochs)

    self.Optimizer(training_data, epochs, mini_batch_size, eta, gamma, optimizer, mode, shuffle, test_data, task)
      
    if test_data:
        self.logging(test_data)

def loss_function(name, y, y_hat, derivative=False):
  """
  Computes the loss and its derivative

  Parameters
  ----------
  name: str
      Type of loss function.
      Options:
          mse ( Mean squared error )
          ce ( Cross entropy ) 

  y: list 
      List of numpy arrays ( target )
  
  y_hat: list
      List of numpy arrays ( output )

  derivative: bool
      If true, returns the derivative of loss.
      Default: False

  Returns
  -------
  List
  """

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
  """
  Computes the activation function and its derivative.

  Parameters
  ----------
  name: str
      Activation function name.
        Options:
            identity
            sigmoid
            softmax
            tanh
            relu  

  input: int/float/list

  derivative: bool
      If true, returns the derivative of loss.
      Default: False

  Returns
  -------
  None
  """

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

