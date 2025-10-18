import numpy as np

class Adam:
    def __init__(self, learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8):  # Reduced learning rate
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0
        
    def update(self, w, grad_w):
        if self.m is None:
            self.m = np.zeros_like(w)
            self.v = np.zeros_like(w)
            
        self.t += 1
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_w
        self.v = self.beta2 * self.v + (1 - self.beta2) * np.square(grad_w)
        
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        
        return w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)