import numpy as np
import pickle
import os
from convlayers import Conv2D
from ann_layers.dense import Dense
from ann_layers.dropout import Dropout
from pooling import MaxPool2D
from utils.activation import ReLU, Softmax
from flatten import Flatten
from tqdm import tqdm


class CNN:
    def __init__(self, input_shape, learning_rate=0.001):
        self.layers = []
        self.learning_rate = learning_rate
        self.input_shape = input_shape
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
    def add(self, layer):
        self.layers.append(layer)
        
    def forward(self, X):
        current_output = X
        for i, layer in enumerate(self.layers):
            try:
                current_output = layer.forward(current_output)
            except Exception as e:
                print(f"Error in layer {i} ({layer.__class__.__name__})")
                print(f"Input shape: {current_output.shape}")
                raise e
        return current_output
        
    def backward(self, grad_output):
        current_grad = grad_output
        for layer in reversed(self.layers):
            current_grad = layer.backward(current_grad, self.learning_rate)
            
    def _compute_metrics_batch(self, X, y, batch_size=32):
        n_samples = len(X)
        total_loss = 0
        correct_predictions = 0
        
        for i in range(0, n_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            predictions = self.forward(batch_X)
            total_loss += -np.sum(np.log(predictions[range(len(batch_y)), batch_y] + 1e-8))
            correct_predictions += np.sum(np.argmax(predictions, axis=1) == batch_y)
            
        avg_loss = total_loss / n_samples
        accuracy = correct_predictions / n_samples
        return avg_loss, accuracy
            
    def train(self, X_train, y_train, X_val, y_val, epochs=30, batch_size=32):
        # Convert data to float32
        X_train = X_train.astype(np.float32)
        X_val = X_val.astype(np.float32)
        
        # Ensure input has correct shape
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(-1, *self.input_shape)
        if len(X_val.shape) == 3:
            X_val = X_val.reshape(-1, *self.input_shape)
            
        print(f"Training data shape: {X_train.shape}")
        print(f"Validation data shape: {X_val.shape}")
        
        n_samples = len(X_train)
        best_val_acc = 0
        patience = 5
        patience_counter = 0
        
        # Learning rate schedule with warmup
        initial_lr = self.learning_rate
        warmup_epochs = 2
        min_lr = 1e-6
        
        for epoch in range(epochs):
            # Learning rate scheduling
            if epoch < warmup_epochs:
                # Gradual warmup
                self.learning_rate = initial_lr * ((epoch + 1) / warmup_epochs)
            else:
                # Cosine decay with minimum learning rate
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                self.learning_rate = min_lr + 0.5 * (initial_lr - min_lr) * (1 + np.cos(progress * np.pi))

            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            batch_losses = []
            running_loss = 0.0  # For smoothing the loss display
            
            n_batches = (n_samples + batch_size - 1) // batch_size
            batch_iterator = tqdm(range(n_batches), desc=f'Epoch {epoch}/{epochs}')
            
            for i in batch_iterator:
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, n_samples)
                
                batch_X = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                # Forward pass
                predictions = self.forward(batch_X)
                
                # Compute loss with label smoothing
                smooth_factor = 0.1
                n_classes = predictions.shape[1]
                smooth_labels = np.zeros_like(predictions)
                smooth_labels[range(len(batch_y)), batch_y] = 1.0
                smooth_labels = smooth_labels * (1 - smooth_factor) + (smooth_factor / n_classes)
                
                batch_loss = -np.mean(np.sum(smooth_labels * np.log(predictions + 1e-8), axis=1))
                running_loss = 0.9 * running_loss + 0.1 * batch_loss if i > 0 else batch_loss
                batch_losses.append(batch_loss)
                
                # Compute gradients
                grad_output = predictions - smooth_labels
                
                # Gradient clipping with norm
                grad_norm = np.linalg.norm(grad_output)
                clip_threshold = 1.0
                if grad_norm > clip_threshold:
                    grad_output = grad_output * (clip_threshold / grad_norm)
                
                # Backward pass
                self.backward(grad_output)
                
                # Update progress bar with smoothed loss
                batch_iterator.set_postfix({
                    'loss': f'{running_loss:.4f}',
                    'lr': f'{self.learning_rate:.6f}'
                })
            
            # Compute epoch metrics
            train_loss, train_acc = self._compute_metrics_batch(X_train, y_train, batch_size)
            val_loss, val_acc = self._compute_metrics_batch(X_val, y_val, batch_size)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print("\nEarly stopping triggered!")
                break
    
    def _update_history(self, train_loss, train_acc, val_loss, val_acc):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)

    def predict(self, X, batch_size=32):
        if len(X.shape) == 3:
            X = X.reshape(-1, *self.input_shape)
        predictions = []
        for i in range(0, len(X), batch_size):
            batch_X = X[i:i+batch_size]
            batch_pred = self.forward(batch_X)
            predictions.append(np.argmax(batch_pred, axis=1))
        return np.concatenate(predictions)

    def save_model(self, filepath):
        """Save the model parameters and architecture to a file"""
        model_data = {
            'input_shape': self.input_shape,
            'learning_rate': self.learning_rate,
            'layers': []
        }
        
        # Save parameters for each layer
        for layer in self.layers:
            layer_data = {
                'type': layer.__class__.__name__,
                'params': {}
            }
            
            # Save layer-specific parameters
            if isinstance(layer, Conv2D):
                layer_data['params'].update({
                    'weights': layer.weights,
                    'bias': layer.bias,
                    'num_filters': layer.num_filters,
                    'kernel_size': layer.kernel_size,
                    'stride': layer.stride,
                    'padding': layer.padding
                })
            elif isinstance(layer, Dense):
                layer_data['params'].update({
                    'weights': layer.weights,
                    'bias': layer.bias
                })
            elif isinstance(layer, Dropout):
                layer_data['params'].update({
                    'p': layer.p
                })
            elif isinstance(layer, MaxPool2D):
                layer_data['params'].update({
                    'pool_size': layer.pool_size,
                    'stride': layer.stride
                })
            
            model_data['layers'].append(layer_data)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
    @classmethod
    def load_model(cls, filepath):
        """Load a saved model from a file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new model instance
        model = cls(model_data['input_shape'], learning_rate=model_data['learning_rate'])
        
        # Reconstruct layers
        for layer_data in model_data['layers']:
            layer_type = layer_data['type']
            params = layer_data['params']
            
            if layer_type == 'Conv2D':
                layer = Conv2D(
                    num_filters=params['num_filters'],
                    kernel_size=params['kernel_size'],
                    stride=params['stride'],
                    padding=params['padding']
                )
                layer.weights = params['weights']
                layer.bias = params['bias']
                
            elif layer_type == 'Dense':
                input_size = params['weights'].shape[0]
                output_size = params['weights'].shape[1]
                layer = Dense(input_size, output_size)
                layer.weights = params['weights']
                layer.bias = params['bias']
                
            elif layer_type == 'Dropout':
                layer = Dropout(p=params['p'])
                
            elif layer_type == 'MaxPool2D':
                layer = MaxPool2D(
                    pool_size=params['pool_size'],
                    stride=params['stride']
                )
                
            elif layer_type in ['ReLU', 'Softmax', 'Flatten']:
                layer = globals()[layer_type]()
                
            model.add(layer)
            
        return model
