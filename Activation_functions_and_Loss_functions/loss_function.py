import numpy as np

def loss_function(name, y, y_hat, derivative=False):
  # y - target, y_hat - output
  if name == "mse":
    if derivative:
      return sum([ y_hat[i] - y[i] for i in range(len(y)) ])
    else:
      return sum([(y_hat[i] - y[i])**2 for i in range(len(y))])/ len(y)

  # y - target prob distro, y_hat - output prob distro
  # Cross Entropy
  if name == "ce":
    if derivative:
        # if activation fn is sigmoid/softmax
        return sum([ y_hat[i] - y[i] for i in range(len(y)) ])      
    else:
        return -sum([y[i]*np.log(y_hat[i]) for i in range(len(y))])
