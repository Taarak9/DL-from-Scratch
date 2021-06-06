import numpy as np

def weight_initializer(NN, name="random"):
    """
    Initializes weights and biases
    
    Parameters
    ----------
    NN: Neural net object
        contains sizes, weights, biases
        
    name: str
        Type of weight initialization.
        Options:
            random ( Gauss distro mean 0, std 1 )
            xavier ( n^2 = 1 / n )
            he ( n^2 = 2 / n )
    
    Returns
    -------
    NN: Neural net object
    """
    
    if name == "random":
        NN.weights = [np.random.randn(y, x) 
                      for x, y in zip(NN.sizes[:-1], NN.sizes[1:])]
        NN.biases = [np.random.randn(y, 1) for y in NN.sizes[1:]]
     
    elif name == "xavier":
        NN.weights = [np.random.randn(y, x)/np.sqrt(x) 
                      for x, y in zip(NN.sizes[:-1], NN.sizes[1:])]
        NN.biases = [np.random.randn(y, 1) for y in NN.sizes[1:]]
    
    elif name == "he":
        NN.weights = [np.random.randn(y, x)*np.sqrt(2/x)
                      for x, y in zip(NN.sizes[:-1], NN.sizes[1:])]
        NN.biases = [np.random.randn(y, 1) for y in NN.sizes[1:]]
        
    return NN