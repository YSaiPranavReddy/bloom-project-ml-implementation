import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        
    def forward(self, input):
        if self.p > 0:
            self.mask = np.random.binomial(1, 1-self.p, input.shape) / (1-self.p)
            return input * self.mask
        return input
        
    def backward(self, grad_output, learning_rate):
        if self.p > 0:
            return grad_output * self.mask
        return grad_output