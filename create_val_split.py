import os
import shutil
import random
import sys
from pathlib import Path

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Config
DATA_DIR = r"c:\Users\userw\Desktop\مهند\شلبي\Datasets\real fake images"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
SPLIT_RATIO = 0.2  # 20% for validation

def create_validation_split():
    if not os.path.exists(TRAIN_DIR):
        print(f"Error: Train directory not found at {TRAIN_DIR}")
        return

    # Create val directory if it doesn't exist
    if not os.path.exists(VAL_DIR):
        os.makedirs(VAL_DIR)
        print(f"Created validation directory: {VAL_DIR}")

    # Process each class
    classes = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    
    for class_name in classes:
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VAL_DIR, class_name)
        
        # Create class directory in val
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)
        
        # Get all images
        images = [f for f in os.listdir(train_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Calculate how many to move
        num_to_move = int(len(images) * SPLIT_RATIO)
        
        if num_to_move == 0:
            print(f"Skipping {class_name}: Not enough images to split.")
            continue
            
        # Randomly select images
        images_to_move = random.sample(images, num_to_move)
        
        print(f"Moving {num_to_move} images from {class_name} to validation set...")
        
        for img in images_to_move:
            src = os.path.join(train_class_dir, img)
            dst = os.path.join(val_class_dir, img)
            shutil.move(src, dst)
            
    print("✅ Validation split creation complete!")

if __name__ == "__main__":
    create_validation_split()
