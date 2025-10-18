import numpy as np
import cv2

class DataAugmenter:
    def __init__(self):
        self.augmentation_methods = [
            self.random_flip,
            self.random_rotation,
            self.random_brightness,
            self.random_contrast
        ]
    
    def augment(self, image):
        """Apply random augmentation to the image"""
        # Randomly select an augmentation method
        method = np.random.choice(self.augmentation_methods)
        return method(image)
    
    def random_flip(self, image):
        """Randomly flip the image horizontally"""
        if np.random.random() > 0.5:
            return np.flip(image, axis=2)  # Flip along width dimension
        return image
    
    def random_rotation(self, image):
        """Randomly rotate the image by 90, 180, or 270 degrees"""
        k = np.random.randint(1, 4)  # Number of 90-degree rotations
        # Transpose back to HWC, rotate, then transpose back to CHW
        image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
        image = np.rot90(image, k)
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return image
    
    def random_brightness(self, image):
        """Randomly adjust brightness"""
        factor = np.random.uniform(0.8, 1.2)
        image = image * factor
        return np.clip(image, -1.0, 1.0)  # Clip values to valid range
    
    def random_contrast(self, image):
        """Randomly adjust contrast"""
        mean = np.mean(image, axis=(1, 2), keepdims=True)
        factor = np.random.uniform(0.8, 1.2)
        image = (image - mean) * factor + mean
        return np.clip(image, -1.0, 1.0)  # Clip values to valid range