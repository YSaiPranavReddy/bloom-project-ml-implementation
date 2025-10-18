import numpy as np

class Dense:
    def __init__(self, input_size, output_size, l2_lambda=0.01):
        # He initialization
        self.weights = np.random.randn(input_size, output_size).astype(np.float32) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros(output_size, dtype=np.float32)
        self.l2_lambda = l2_lambda
        
        # Adam optimizer parameters
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
    def forward(self, input):
        self.input = input
        # Weight normalization for better stability
        self.weight_norm = np.linalg.norm(self.weights, axis=0, keepdims=True)
        self.normalized_weights = self.weights / (self.weight_norm + self.epsilon)
        return np.dot(input, self.normalized_weights) + self.bias
        
    def backward(self, grad_output, learning_rate):
        self.t += 1
        
        grad_input = np.dot(grad_output, self.weights.T)
        
        # Compute gradients with L2 regularization and weight normalization
        grad_weights = (np.dot(self.input.T, grad_output) + 
                       self.l2_lambda * self.normalized_weights)
        grad_bias = np.sum(grad_output, axis=0)
        
        # Applying gradient clipping
        grad_norm = np.linalg.norm(grad_weights)
        clip_threshold = 1.0
        if grad_norm > clip_threshold:
            grad_weights = grad_weights * (clip_threshold / grad_norm)
        
        # Adam updates for weights
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * grad_weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * np.square(grad_weights)
        m_hat = self.m_weights / (1 - self.beta1**self.t)
        v_hat = self.v_weights / (1 - self.beta2**self.t)
        self.weights -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Adam updates for bias 
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * grad_bias
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * np.square(grad_bias)
        m_hat_bias = self.m_bias / (1 - self.beta1**self.t)
        v_hat_bias = self.v_bias / (1 - self.beta2**self.t)
        self.bias -= learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
        
        return grad_input
