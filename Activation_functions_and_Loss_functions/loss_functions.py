import numpy as np

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
      numpy array ( target )
  
  y_hat: list
      numpy array ( output )

  derivative: bool
      If True, returns the derivative of loss.
      Default: False

  Returns
  -------
  numpy array
  """
  
  # y - target, y_hat - output
  # Mean Squared Error
  if name == "mse":
      if derivative:
          return (y_hat - y)
      else:
          return np.mean((y - y_hat)**2)
  # Log-likelihood
  elif name == "ll":
      if derivative:
          return - (1 / y_hat)
      else:
          return -1 * np.log(y_hat)
  # y - target prob distro, y_hat - output prob distro
  # Cross Entropy
  elif name == "ce":
      if derivative:
          # if activation fn is sigmoid/softmax
          return (y_hat - y)    
      else:
          return np.sum(np.nan_to_num(-y*np.log(y_hat)-(1-y)*np.log(1-y_hat)))