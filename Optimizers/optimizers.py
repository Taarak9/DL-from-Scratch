import numpy as np
from generalized_fnn import FNN

class Optimizers:

  def __init__(self):
    self.weights, self.biases = FNN.get_params()
    self.prev_update_w = list()
    self.prev_update_b = list()

  def get_params(self):
    return [self.weights, self.biases]

  def get_batch_size(self, training_data, mode, mini_batch_size):
    if mode == "online":
        return 1
    elif mode == "mini_batch":
        return mini_batch_size
    elif mode == "batch":
        return len(training_data)

  def update_GD(self, mini_batch, eta):
    gradient_w = [np.zeros(w.shape) for w in self.weights]
    gradient_b = [np.zeros(b.shape) for b in self.biases]
    for x, y in mini_batch:
        delta_gradient_w, delta_gradient_b = FNN.backprop(x, y)
        gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]
        gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]

    self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, gradient_b)]
    self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, gradient_w)]

  def update_MGD(self, mini_batch, eta, gamma):
    gradient_w = [np.zeros(w.shape) for w in self.weights]
    gradient_b = [np.zeros(b.shape) for b in self.biases]
    for x, y in mini_batch:
        delta_gradient_w, delta_gradient_b = FNN.backprop(x, y)
        gradient_w = [nw + dnw for nw, dnw in zip(gradient_w, delta_gradient_w)]
        gradient_b = [nb + dnb for nb, dnb in zip(gradient_b, delta_gradient_b)]

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
  optimizer: GD ( Gradient Descent), MGD ( Momentum based GD ), NAG ( Nesterov Acceralated GD )
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

