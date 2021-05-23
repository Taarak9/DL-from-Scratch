import numpy as np

def activation_function(self, name, input, derivative=False):
  if name == "identity":
      if derivative:
          return np.ones_like(x)
      else:
          return x
  elif name == "sigmoid":
      if derivative:
          out = self.activation_function(name, input)
          return out * ( 1 - out )
      else:
          return 1 / (1 + np.exp(-input))
  elif name == "softmax":
      if derivative:
          out = self.activation_function(name, input)
          return out * (1 - out)
      else:
          e_x = np.exp(input - np.max(input))
          return e_x / np.sum(e_x, axis=1, keepdims=True)
  elif name == "tanh":
      if derivative:
          out = self.activation_function(name, input)
          return 1 - np.square(out)
      else:
          return 2 / (1 + np.exp(-2*input)) - 1
  elif name == "relu":
      if derivative:
          return (input > 0) * 1
      else:
          return np.maximum(0, input)
