import numpy as np

class Conv2D:
    def __init__(self, num_filters, kernel_size, stride=1, padding=0):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = None
        self.bias = None
        # Adam optimizer parameters
        self.m_weights = None
        self.v_weights = None
        self.m_bias = None
        self.v_bias = None
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
    def initialize(self, input_channels):
        # He initialization
        scale = np.sqrt(2.0 / (input_channels * self.kernel_size * self.kernel_size))
        self.weights = np.random.randn(
            self.num_filters, input_channels, self.kernel_size, self.kernel_size
        ).astype(np.float32) * scale
        self.bias = np.zeros(self.num_filters, dtype=np.float32)
        
        # Initialize Adam parameters
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        
    def im2col(self, input_data, h_filter, w_filter, stride):
        N, C, H, W = input_data.shape
        out_h = (H + 2 * self.padding - h_filter) // stride + 1
        out_w = (W + 2 * self.padding - w_filter) // stride + 1

        if self.padding > 0:
            input_data = np.pad(input_data, 
                              ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)),
                              mode='constant')

        col = np.zeros((N, C, h_filter, w_filter, out_h, out_w), dtype=np.float32)
        
        for y in range(h_filter):
            y_max = y + stride * out_h
            for x in range(w_filter):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = input_data[:, :, y:y_max:stride, x:x_max:stride]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col

    def col2im(self, col, input_shape, h_filter, w_filter, stride):
        N, C, H, W = input_shape
        out_h = (H + 2 * self.padding - h_filter) // stride + 1
        out_w = (W + 2 * self.padding - w_filter) // stride + 1
        
        img = np.zeros((N, C, H + 2 * self.padding, W + 2 * self.padding), dtype=np.float32)
        col = col.reshape(N, out_h, out_w, C, h_filter, w_filter).transpose(0, 3, 4, 5, 1, 2)
        
        for y in range(h_filter):
            y_max = y + stride * out_h
            for x in range(w_filter):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

        if self.padding > 0:
            img = img[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return img

    def forward(self, input):
        if self.weights is None:
            self.initialize(input.shape[1])
            
        self.input = input.astype(np.float32)
        batch_size, channels, height, width = input.shape
        
        out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        self.col = self.im2col(input, self.kernel_size, self.kernel_size, self.stride)
        self.col_W = self.weights.reshape(self.num_filters, -1).T
        
        out = np.dot(self.col, self.col_W) + self.bias
        out = out.reshape(batch_size, out_h, out_w, self.num_filters)
        out = out.transpose(0, 3, 1, 2)
        
        return out
        
    def backward(self, grad_output, learning_rate):
        batch_size, num_filters, out_h, out_w = grad_output.shape
        
        grad_output = grad_output.transpose(0, 2, 3, 1).reshape(-1, self.num_filters)
        
        grad_weights = np.dot(self.col.T, grad_output)
        grad_weights = grad_weights.transpose(1, 0)
        grad_weights = grad_weights.reshape(self.weights.shape)
        
        grad_bias = np.sum(grad_output, axis=0)
        
        grad_col = np.dot(grad_output, self.col_W.T)
        grad_input = self.col2im(grad_col, self.input.shape,
                               self.kernel_size, self.kernel_size, self.stride)
        
        # Adam optimization
        self.t += 1
        
        # Update weights
        self.m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * grad_weights
        self.v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * np.square(grad_weights)
        m_hat = self.m_weights / (1 - self.beta1**self.t)
        v_hat = self.v_weights / (1 - self.beta2**self.t)
        
        # Apply gradient clipping
        grad_norm = np.linalg.norm(grad_weights)
        clip_threshold = 1.0
        if grad_norm > clip_threshold:
            grad_weights = grad_weights * (clip_threshold / grad_norm)
            
        self.weights -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # Update bias
        self.m_bias = self.beta1 * self.m_bias + (1 - self.beta1) * grad_bias
        self.v_bias = self.beta2 * self.v_bias + (1 - self.beta2) * np.square(grad_bias)
        m_hat_bias = self.m_bias / (1 - self.beta1**self.t)
        v_hat_bias = self.v_bias / (1 - self.beta2**self.t)
        self.bias -= learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)
        
        return grad_input
