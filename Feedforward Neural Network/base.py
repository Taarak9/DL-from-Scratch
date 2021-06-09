import numpy as np

def activation_function(name, x, derivative=False):
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
    
    x: int/float/list/array
        Input.
    
    derivative: bool
        If true, returns the derivative of loss.
        Default: False
    
    Returns
    -------
    Numpy array or list
    """
    
    if name == "identity":
        if derivative:
            return np.ones_like(x)
        else:
            return x
    elif name == "sigmoid":
        if derivative:
            out = activation_function(name, x)
            return out * ( 1 - out )
        else:
            # Prevents overflow
            x_clipped = np.clip(x, -500, 500)
            return 1 / (1 + np.exp(-x_clipped))
    elif name == "softmax":
        if derivative:
            out = activation_function(name, x)
            return out * (1 - out)
        else:
            # Prevents overflow
            x_clipped = np.clip(x, -500, 500)
            e_x = np.exp(x_clipped - np.max(x_clipped))
            return e_x / np.sum(e_x, axis=1, keepdims=True)
    elif name == "tanh":
        if derivative:
            out = activation_function(name, x)
            return 1 - np.square(out)
        else:
            return 2 / (1 + np.exp(-2*x)) - 1
    elif name == "relu":
        if derivative:
            return (x > 0) * 1
        else:
            return np.maximum(0, x)
      
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
            # prevents overflow
            y_clipped = np.clip(y_hat, 1e-8, None)
            return np.sum(np.nan_to_num(-y*np.log(y_clipped)-(1-y)*
                                        np.log(1-y_clipped)))