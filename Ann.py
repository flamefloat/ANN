import numpy as np 

class Linear():
    def __init__(self, input_features, output_features):
        self.input_features = input_features
        self.output_features = output_features
        self.weights = np.zeros((self.input_features, self.output_features))

    def forward(self, input_x):
        x_batch = input_x.shape[0]
        input_x = input_x.T
        out = np.dot(input_x, self.weights) # X^T * W
        return out

    def backward(self, grad):
        self.weights = self.weights - grad

class Function():
    def __init__(self):

    def forward(self, input_x):
        return 1/(1 + np.exp(-input_x)) # sigmoid

    def backward(self, input_x):
        return ( 1/(1 + np.exp(-input_x)) ) * ( 1- 1/(1 + np.exp(-input_x)) ) # grad(sigmoid) = sigmoid(1-sigmoid)




class ANN():
    def __init__(self,):
