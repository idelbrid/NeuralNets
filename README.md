# NeuralNets

This repository is contains a feed forward neural network written in python. The code is not designed for production usage - other languages are much better suited. Additionally, there is code for reading MNIST hand writing digits to demonstrate usage.

Future work might be to include more kinds of regularization like dropout and to implement in a faster language, possibly utilizing Cython.

Available features of the Neural Nets are:
* Arbitrary size neural network
* Cross entropy or mean squared error cost function
* Use stochastic gradient descent with arbitrary mini-batch size (gradient descent using largest batch size, on-line SGD with smallest batch size)
* Adjustable learning rate function (later implementing a time-dependent rate)
* L2 weight decay with adjustable regularization parameter
