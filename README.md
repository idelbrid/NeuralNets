# NeuralNets

This repository contains two major code bases. The first is from-scratch python implementations of feed forward neural networks which I originally used to cement the ideas of backprop and different kinds of loss and regularization. The second code base is a shared-weight multiple instance inspired model bootstrapped from auto-encoder training, designed using TensorFlow. This unit is being written as of 5/08/17.

The code is not designed for production usage - it is academic. The python feed forward models were developed while I was reading http://neuralnetworksanddeeplearning.com/chap1.html. Note that other languages are much better suited for performance.

Future work might be to include more kinds of regularization like dropout and to implement in a faster language, possibly utilizing Cython. Alternatively other languages may be utilized, e.g. Julia or C. 

Available features of the Neural Nets are:
* Arbitrary size neural network (depth and layer sizes)
* Cross entropy or mean squared error cost function
* Use stochastic gradient descent with arbitrary mini-batch size (gradient descent using largest batch size, on-line SGD with smallest batch size)
* Adjustable learning rate function (later implementing a time-dependent rate)
* L2 weight decay with adjustable regularization parameter
