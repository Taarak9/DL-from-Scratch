import numpy as np

# y and y_hat are numpy arrays
def loss_function(name, y, y_hat, derivative=False):
  # y - target, y_hat - output
  # Mean Squared Error
  if name == "mse":
      if derivative:
          return 
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
