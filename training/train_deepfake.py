import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
import copy
import sys
import matplotlib.pyplot as plt

# Force UTF-8 encoding for stdout
sys.stdout.reconfigure(encoding='utf-8')

# Configuration
DATA_DIR = r"c:\Users\userw\Desktop\مهند\شلبي\Datasets\real fake images"
MODEL_SAVE_PATH = r"c:\Users\userw\Desktop\مهند\شلبي\backend\models\deepfake_model.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 2  # Reduced for fast demo training
LEARNING_RATE = 0.001
IMG_SIZE = 224
MAX_IMAGES_PER_CLASS = 2500 # Limit data for speed on CPU

def train_model():
    print("🚀 Starting Optimized Deepfake Detection Training...")
    
    # 1. Data Augmentation & Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Load Data with Subsetting
    image_datasets = {}
    phases = ['train', 'val', 'test']
    
    for x in phases:
        path = os.path.join(DATA_DIR, x)
        if not os.path.exists(path):
            print(f"⚠️ Warning: {path} not found. Skipping.")
            continue
        
        full_dataset = datasets.ImageFolder(path, data_transforms[x])
        
        # Subsetting logic: take only first N images per class
        indices = []
        class_counts = {i: 0 for i in range(len(full_dataset.classes))}
        for idx, (_, label) in enumerate(full_dataset.samples):
            if class_counts[label] < MAX_IMAGES_PER_CLASS:
                indices.append(idx)
                class_counts[label] += 1
        
        image_datasets[x] = torch.utils.data.Subset(full_dataset, indices)
        # Add classes attribute back for convenience
        image_datasets[x].classes = full_dataset.classes

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) 
                  for x in image_datasets}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets}
    class_names = image_datasets['train'].classes
    
    print(f"✅ Data loaded (Subsampled): {dataset_sizes}")
    print(f"✅ Classes: {class_names}")

    # 3. Setup Model (MobileNetV2)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"💻 Training on: {device}")

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze feature layers (Transfer Learning)
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace Classifier Head for 2 classes (Real vs Fake)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, 2) # Binary Classification
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # History for plotting
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(NUM_EPOCHS):
        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase not in dataloaders: continue

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 5. Save Model
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"💾 Model saved to {MODEL_SAVE_PATH}")
    
    # 6. Plot Curves
    plot_curves(history)

def plot_curves(history):
    # Accuracy Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    print("📊 Accuracy curve saved as accuracy_curve.png")
    
    # Loss Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')
    print("📊 Loss curve saved as loss_curve.png")

if __name__ == '__main__':
    train_model()
