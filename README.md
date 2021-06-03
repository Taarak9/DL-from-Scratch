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

## Feedforward Neural Network   . 
* fnn.py - Generic Feedforward Neural Network.
* [customdl](https://pypi.org/project/customdl/) package
* [README](https://github.com/Taarak9/Neural-Networks/tree/master/Feedforward%20Neural%20Network)

## Linear Classifier          
Aeroplane Classification using Linear Classifier with two variants
* Perceptron learning rule; mode = online learning, loss fn =  MSE
* Gradient descent; mode = batch learning, loss fn = Cross Entropy

## Hopfield Neural Network   
Distorted Character Recognition using Hopfield Neural Network                
                         
`To get started with Neural Networks I recommend the [playlist](https://youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) by 3Blue1Brown.`

### To-do list
* [ ] Use validation data for parameter selection
* [ ] Add optimizers: Adam and RMSProp
* [ ] Write seperate fn for weight initialization methods
* [ ] Add regularization techniques: L1, L2, dropout
