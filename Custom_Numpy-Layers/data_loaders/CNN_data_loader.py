import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from data_augumentation import DataAugmenter

def load_and_preprocess_image(image_path, target_size=(128, 128), augmenter=None):
    try:
        # Read image in RGB mode
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Failed to load image")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        
        # Convert to float32 and normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Global normalization
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        
        # Transpose to (channels, height, width)
        img = img.transpose(2, 0, 1)
        
        # Apply augmentation if provided
        if augmenter is not None:
            img = augmenter.augment(img)
        
        return img
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def load_dataset(folder_path, augment=False, augment_factor=2):
    print("Loading dataset...")
    images = []
    labels = []
    
    # Get list of class directories
    class_dirs = [d for d in sorted(os.listdir(folder_path)) 
                 if os.path.isdir(os.path.join(folder_path, d))]
    
    # Initialize augmenter if needed
    augmenter = DataAugmenter() if augment else None
    
    for class_idx, class_name in enumerate(class_dirs):
        class_path = os.path.join(folder_path, class_name)
        print(f"Loading class {class_name}...")
        
        # Get list of image files
        valid_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = [f for f in os.listdir(class_path) 
                      if os.path.splitext(f)[1].lower() in valid_extensions]
        
        class_images = []
        class_labels = []
        
        for img_name in image_files:
            img_path = os.path.join(class_path, img_name)
            try:
                # Load and preprocess original image
                img_array = load_and_preprocess_image(img_path)
                if img_array is not None:
                    class_images.append(img_array)
                    class_labels.append(class_idx)
                    
                    # Generate augmented versions if needed
                    if augment:
                        for _ in range(augment_factor - 1):
                            aug_img = load_and_preprocess_image(img_path, augmenter=augmenter)
                            if aug_img is not None:
                                class_images.append(aug_img)
                                class_labels.append(class_idx)
                                
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        images.extend(class_images)
        labels.extend(class_labels)
        
        print(f"Loaded {len(class_images)} images for class {class_name}")
    
    if not images:
        raise Exception("No valid images were loaded")
    
    # Convert to float32 arrays
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    print(f"Total loaded {len(images)} images from {len(class_dirs)} classes")
    print(f"Final data shape: {X.shape}")
    
    return X, y