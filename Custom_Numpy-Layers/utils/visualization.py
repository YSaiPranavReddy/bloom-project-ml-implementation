import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_training_history(history):
    """Plot training and validation loss/accuracy."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def display_sample_predictions(X, y_true, y_pred, class_names, samples=5):
    """Display sample images with their true and predicted labels."""
    plt.figure(figsize=(15, 3))
    indices = np.random.choice(len(X), samples, replace=False)
    
    for idx, i in enumerate(indices):
        plt.subplot(1, samples, idx + 1)
        # Convert from (channels, height, width) to (height, width, channels)
        img = X[i].transpose(1, 2, 0)
        # Normalize to [0, 1] range for display
        img = (img - img.min()) / (img.max() - img.min())
        plt.imshow(img)
        plt.title(f'True: {class_names[y_true[i]]}\nPred: {class_names[y_pred[i]]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()