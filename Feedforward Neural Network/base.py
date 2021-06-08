import numpy as np

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
    
    input: int/float/list/array
    
    derivative: bool
        If true, returns the derivative of loss.
        Default: False
    
    Returns
    -------
    Numpy array or list
    """
    
    if name == "identity":
        if derivative:
            return np.ones_like(input)
        else:
            return input
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