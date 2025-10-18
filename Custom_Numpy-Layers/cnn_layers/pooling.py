import numpy as np

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.cache = None
        
    def forward(self, input):
        self.input = input
        batch_size, channels, height, width = input.shape
        
        # Calculate output dimensions
        out_height = (height - self.pool_size) // self.stride + 1
        out_width = (width - self.pool_size) // self.stride + 1
        
        # Initialize output and cache for max locations
        output = np.zeros((batch_size, channels, out_height, out_width))
        self.cache = np.zeros_like(input)
        
        # Perform max pooling
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                # Get current window
                window = input[:, :, h_start:h_end, w_start:w_end]
                
                # Find maximum values and their locations
                output[:, :, i, j] = np.max(window, axis=(2, 3))
                
                # Store locations of maximum values for backpropagation
                max_mask = (window == output[:, :, i, j][:, :, None, None])
                self.cache[:, :, h_start:h_end, w_start:w_end] += max_mask
                
        return output
        
    def backward(self, grad_output, learning_rate=None):
        batch_size, channels, out_height, out_width = grad_output.shape
        grad_input = np.zeros_like(self.input)
        
        # Distribute gradients to max locations
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                
                # Get mask for this window
                mask = self.cache[:, :, h_start:h_end, w_start:w_end]
                
                # Distribute gradient to max locations
                grad_input[:, :, h_start:h_end, w_start:w_end] += \
                    mask * grad_output[:, :, i, j][:, :, None, None]
                
        return grad_input