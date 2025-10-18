import numpy as np

class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)
        
    def backward(self, grad_output, learning_rate):
        return grad_output * (self.input > 0)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        
    def forward(self, input):
        self.input = input
        return np.where(input > 0, input, input * self.alpha)
        
    def backward(self, grad_output, learning_rate):
        return np.where(self.input > 0, grad_output, grad_output * self.alpha)

class Tanh:
    def forward(self, input):
        self.output = np.tanh(input)
        return self.output
        
    def backward(self, grad_output, learning_rate):
        return grad_output * (1 - self.output ** 2)

class Softmax:
    def forward(self, input):
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output
        
    def backward(self, grad_output, learning_rate):
        return grad_output