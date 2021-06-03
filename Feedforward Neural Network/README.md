# Generic Feedforward Neural Network   
## Initialization
* Initialize the NN parameters such as input nodes, number of hidden layers (number of hidden nodes and activation function used), loss function, optiimizer, learning mode etc.
* Initialze the weights and biases using weight initilization techniques.

## Training
* Feed all the training examples to the NN and compute the outputs ( Feedforward ). 
* The Loss value quantifies the deviation between our predicted output and the target.
* Aim of our training algorithm is to minimize this loss by tuning the weights and biases.
* Optimizer ( Ex: Stochastic mini-batch Gradient Descent ) helps us by giving a mechanism for updating the weights and biases based on the gradients of loss w.r.t weights and biases. ( Gradient descent is an optimization algoritm which tells us the direction to roll the ball ( weights and biases ) to reach the minima of the Loss surface )
* We use backpropagation to computes these gradients.  
* In each epoch, we randomly shuffling the training data, and then partition it into mini-batches. 
   * Then for each mini-batch we apply a single step of gradient descent, which updates the weights and biases.
* We do the above step until the convergence or for some fixed number of epochs. 

## Validation and Testing
* For the epochs we ran, we could save the NN configuration at the epoch where it has the minimum loss :)
* And with this NN configuration we test our predictions on the test data.

* Install the [customdl](https://pypi.org/project/customdl/) package to play with this NN.
