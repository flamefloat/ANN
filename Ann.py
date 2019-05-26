import numpy as np 

class Linear():
    def __init__(self, input_features, output_features, learningRate):
        self.input_features = input_features
        self.output_features = output_features
        self.learningRate = learningRate
        self.weights = np.zeros((self.input_features, self.output_features))

    def forward(self, input_x):
        output_x = np.dot(input_x, self.weights) # X^T * W
        self.input_x = input_x # shape(1, input_features)
        return output_x

    def backward(self, self_grad):
        grad_weigths = np.dot(self.input_x.T, self_grad)
        self.weights = self.weights - self.learningRate * grad_weigths

class Function():
    def __init__(self):
        pass
    def forward(self, input_x):
        sigmoid_x = 1/(1 + np.exp(-input_x))
        return sigmoid_x # sigmoid

    def backward(self, input_x):
        grad_sigmoid_x = ( 1/(1 + np.exp(-input_x)) ) * ( 1- 1/(1 + np.exp(-input_x)) )
        return grad_sigmoid_x # grad(sigmoid) = sigmoid(1-sigmoid)



class ANN():
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learningRate):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learningRate = learningRate
        self.fc1 = Linear(self.input_size, self.hidden_size1, self.learningRate)
        self.fc2 = Linear(self.hidden_size1, self.hidden_size2, self.learningRate)
        self.fc3 = Linear(self.hidden_size2, self.output_size, self.learningRate)

    def forward(self, input_x):
        self.sigmoid = Function()
        fc1_out = self.sigmoid.forward( self.fc1.forward(input_x) )
        fc2_out = self.sigmoid.forward( self.fc2.forward(fc1_out) )
        fc3_out = self.fc3.forward(fc2_out)
        self.fc1_out = fc1_out
        self.fc2_out = fc2_out
        self.fc3_out = fc3_out
        return fc3_out

    def backward(self, label):
        fc3_self_grad = -(label - self.fc3_out) # MSEloss
        self.fc3.backward(fc3_self_grad) # 链式法则求梯度
        fc2_self_grad = np.dot(fc3_self_grad, self.fc3.weights.T) * self.sigmoid.backward(self.fc2_out)
        self.fc2.backward(fc2_self_grad)
        fc1_self_grad = np.dot(fc2_self_grad, self.fc2.weights.T) * self.sigmoid.backward(self.fc1_out)
        self.fc1.backward(fc1_self_grad)


    



