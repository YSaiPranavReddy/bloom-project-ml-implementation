import numpy as np

class BatchNormalization:
    def __init__(self, momentum=0.99, epsilon=1e-8):
        self.momentum = momentum
        self.epsilon = epsilon
        self.gamma = None
        self.beta = None
        self.moving_mean = None
        self.moving_var = None
        self.input_shape = None
        self.training = True
        
    def initialize(self, input_shape):
        self.input_shape = input_shape
        # For CNN, shape should be (channels,) or (channels, 1, 1)
        if len(input_shape) == 4:  # CNN input (batch, channels, height, width)
            channels = input_shape[1]
        elif len(input_shape) == 2:  # Dense layer input (batch, features)
            channels = input_shape[1]
        else:
            raise ValueError(f"Unexpected input shape: {input_shape}")
            
        self.gamma = np.ones(channels, dtype=np.float32)
        self.beta = np.zeros(channels, dtype=np.float32)
        self.moving_mean = np.zeros(channels, dtype=np.float32)
        self.moving_var = np.ones(channels, dtype=np.float32)
        
    def forward(self, input):
        # Ensure input is float32
        input = np.asarray(input, dtype=np.float32)
        
        if self.gamma is None:
            self.initialize(input.shape)
            
        # Store input for backprop
        self.input = input
        self.input_shape = input.shape
        
        # Handle different input shapes
        if len(input.shape) == 4:  # CNN input
            N, C, H, W = input.shape
            # Reshape to 2D: (N*H*W, C)
            input_reshaped = input.transpose(0, 2, 3, 1).reshape(-1, C)
        elif len(input.shape) == 2:  # Dense layer input
            N, C = input.shape
            input_reshaped = input
        else:
            raise ValueError(f"Unexpected input shape: {input.shape}")
        
        if self.training:
            # Calculate mean and variance
            self.mean = np.mean(input_reshaped, axis=0)
            self.var = np.var(input_reshaped, axis=0)
            
            # Update moving statistics
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.mean
            self.moving_var = self.momentum * self.moving_var + (1 - self.momentum) * self.var
        else:
            self.mean = self.moving_mean
            self.var = self.moving_var
        
        # Normalize
        std = np.sqrt(self.var + self.epsilon)
        self.normalized = (input_reshaped - self.mean) / std
        
        # Scale and shift
        output = self.gamma * self.normalized + self.beta
        
        # Reshape output back to original shape
        if len(self.input_shape) == 4:
            output = output.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            
        return output.astype(np.float32)
    
    def backward(self, grad_output, learning_rate):
        grad_output = np.asarray(grad_output, dtype=np.float32)
        
        if len(self.input_shape) == 4:
            N, C, H, W = self.input_shape
            grad_output_reshaped = grad_output.transpose(0, 2, 3, 1).reshape(-1, C)
        else:
            N, C = self.input_shape
            grad_output_reshaped = grad_output
            
        # Compute gradients
        std = np.sqrt(self.var + self.epsilon)
        
        # Gradients with respect to gamma and beta
        self.grad_gamma = np.sum(grad_output_reshaped * self.normalized, axis=0)
        self.grad_beta = np.sum(grad_output_reshaped, axis=0)
        
        # Gradient with respect to normalized input
        grad_normalized = grad_output_reshaped * self.gamma
        
        # Gradient with respect to input
        grad_input = grad_normalized / std
        
        # Update parameters
        self.gamma -= learning_rate * self.grad_gamma
        self.beta -= learning_rate * self.grad_beta
        
        # Reshape gradient back to original shape if needed
        if len(self.input_shape) == 4:
            grad_input = grad_input.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            
        return grad_input