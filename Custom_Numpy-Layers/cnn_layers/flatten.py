import numpy as np

class Flatten:
    def __init__(self):
        self.input_shape = None
        
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)
        
    def backward(self, grad_output, learning_rate=None):
        return grad_output.reshape(self.input_shape)