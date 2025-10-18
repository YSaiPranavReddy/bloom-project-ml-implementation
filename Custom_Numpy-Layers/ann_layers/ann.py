import numpy as np
from dense import Dense
from utils.activation import ReLU, Softmax
from dropout import Dropout

class ANN:
    def __init__(self, input_size, learning_rate=0.001):
        self.layers = []
        self.learning_rate = learning_rate
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, X):
        current_output = X
        for layer in self.layers:
            current_output = layer.forward(current_output)
        return current_output
        
    def backward(self, grad_output):
        current_grad = grad_output
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad, self.learning_rate)
            
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        n_samples = len(X_train)
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]

            learning_rate_schedule_epoch = 15  # Epoch at which to change the learning rate
            lr_decay_factor = 0.1 
            if epoch >= learning_rate_schedule_epoch and epoch%10==0:
                self.learning_rate = self.learning_rate * lr_decay_factor
                print(f"At Epoch {epoch} learning rate = {self.learning_rate} ")

            for i in range(0, n_samples, batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                predictions = self.forward(batch_X)
                
                grad_output = predictions.copy()
                grad_output[range(len(batch_y)), batch_y] -= 1
                grad_output /= len(batch_y)
                
                self.backward(grad_output)
            
            train_loss, train_acc = self._compute_metrics(X_train, y_train)
            val_loss, val_acc = self._compute_metrics(X_val, y_val)
            
            self._update_history(train_loss, train_acc, val_loss, val_acc)
            
            # if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    def _compute_metrics(self, X, y):
        predictions = self.forward(X)
        loss = -np.mean(np.log(predictions[range(len(y)), y] + 1e-8))
        accuracy = np.mean(np.argmax(predictions, axis=1) == y)
        return loss, accuracy
        
    def _update_history(self, train_loss, train_acc, val_loss, val_acc):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

    def predict(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)