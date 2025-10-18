import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_image(image_path, target_size=(64, 64)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = cv2.equalizeHist(img)
    
    # Normalize to [0,1] and reshape for CNN (batch_size, channels, height, width)
    img_array = img.astype(np.float32) / 255.0
    return img_array.reshape(1, target_size[0], target_size[1])

def load_dataset(folder_path):
    images = []
    labels = []
    class_dirs = sorted(os.listdir(folder_path))
    
    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(folder_path, class_name)
        if not os.path.isdir(class_path):
            continue
            
        print(f"Loading class {class_name}...")
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img_array = load_and_preprocess_image(img_path)
                images.append(img_array)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    X = np.array(images)
    y = np.array(labels)
    
    # Standardize features
    batch_size, channels, height, width = X.shape
    X = X.reshape(batch_size, -1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape(batch_size, channels, height, width)
    
    return X, y