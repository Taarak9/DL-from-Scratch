# Neural-Networks

## Activation functions and Loss functions
 * Activation functions and their derivatives
     * Identity, Sigmoid, Softmax, Tanh, ReLU
 * Loss functions and their derivatives
     * Mean Squared Error, Cross Entropy   
 
## Optimizers
* Stochastic mini-batch Gradient Descent
* Momentum based Gradient Descent
* Nesterov accelerated Gradient Descent
* ...

## Feedforward Neural Network   
* Backpropagation ( Computes the gradient of Loss function ) Learning using user-specified optimizer  
    * In each epoch, it starts by randomly shuffling the training data, and then partitions it into mini-batches. 
    * Then for each mini-batch we apply a single step of gradient descent, which updates the network weights and biases. 
* fnn.py - Custom Feedforward Neural Network with user-specified activation functions for layers, optimizer and loss function. 
* [customneuralnet package](https://pypi.org/project/customneuralnet/) is based on fnn.py

## Linear Classifier          
Aeroplane Classification using Linear Classifier with two variants
* Perceptron learning rule; mode = online learning, loss fn =  MSE
* Gradient descent; mode = batch learning, loss fn = Cross Entropy

## Hopfield Neural Network   
Distorted Character Recognition using Hopfield Neural Network                
                         
#### To get started with Neural Networks I recommend the [playlist](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3Blue1Brown.
