"""
Training Script for Plant Disease Classification Model
Uses PlantVillage Dataset with Transfer Learning (MobileNetV2) - PyTorch Version
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os
import json
import time
from tqdm import tqdm
from datetime import datetime

# ============== GPU Check (REQUIRED) ==============
if not torch.cuda.is_available():
    print("=" * 60)
    print("❌ خطأ: لم يتم العثور على GPU!")
    print("=" * 60)
    print("\nهذا السكربت يتطلب GPU للتدريب.")
    print("تأكد من:")
    print("  1. تثبيت CUDA Toolkit")
    print("  2. تثبيت PyTorch مع دعم CUDA:")
    print("     pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    print("\nللتحقق من GPU:")
    print("  nvidia-smi")
    print("=" * 60)
    exit(1)

device = torch.device("cuda")
print(f"🚀 GPU Found: {torch.cuda.get_device_name(0)}")
print(f"   CUDA Version: {torch.version.cuda}")
print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# ============== Configuration ==============
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = {
    'data_dir': os.path.join(BASE_DIR, 'data', 'plantvillage dataset', 'color'),
    'model_save_path': os.path.join(BASE_DIR, 'backend', 'models', 'classifier_model.pth'),
    'history_save_path': os.path.join(BASE_DIR, 'backend', 'models', 'training_history.png'),
    'results_save_path': os.path.join(BASE_DIR, 'backend', 'models', 'training_results.json'),
    'img_size': 224,
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'num_workers': 4,
}

# ============== Data Preparation ==============
def prepare_data():
    """Prepare data loaders for training and validation"""
    
    # Data transforms
    train_transforms = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((CONFIG['img_size'], CONFIG['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(CONFIG['data_dir'], transform=train_transforms)
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Apply val transforms to validation set
    val_dataset.dataset.transform = val_transforms
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )
    
    return train_loader, val_loader, full_dataset.classes

# ============== Model Building ==============
def build_model(num_classes):
    """Build classification model with MobileNetV2 backbone"""
    
    # Load pretrained MobileNetV2
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Freeze base layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, num_classes)
    )
    
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model built with {num_classes} classes")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model

# ============== Training ==============
def train_epoch(model, loader, criterion, optimizer, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} [Train]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return running_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total

def train_model(model, train_loader, val_loader, epochs, lr):
    """Full training loop"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    print("\n🚀 Starting training...")
    print("=" * 50)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, CONFIG['model_save_path'])
            print(f"  ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
                break
    
    total_time = time.time() - start_time
    print("=" * 50)
    print(f"✅ Training completed in {total_time/60:.1f} minutes")
    print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
    
    return history

# ============== Fine-tuning ==============
def fine_tune_model(model, train_loader, val_loader, epochs=10):
    """Fine-tune by unfreezing some layers"""
    print("\n🔧 Fine-tuning model...")
    
    # Unfreeze last few layers
    for param in model.features[-5:].parameters():
        param.requires_grad = True
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters after unfreezing: {trainable:,}")
    
    # Train with lower learning rate
    history = train_model(model, train_loader, val_loader, epochs, lr=CONFIG['learning_rate'] / 10)
    
    return history

# ============== Visualization ==============
def plot_training_history(history, save_path):
    """Plot and save training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_acc']) + 1)
    
    # Accuracy
    axes[0].plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy')
    axes[0].plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(epochs, history['train_loss'], 'b-o', label='Training Loss')
    axes[1].plot(epochs, history['val_loss'], 'r-o', label='Validation Loss')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Training curves saved to: {save_path}")

# ============== Main ==============
def main():
    print("=" * 60)
    print("🌿 Plant Disease Classification Training (PyTorch)")
    print("=" * 60)
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check data directory
    if not os.path.exists(CONFIG['data_dir']):
        print(f"\n❌ Data directory not found: {CONFIG['data_dir']}")
        print("\nPlease download PlantVillage dataset from:")
        print("https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print(f"\nExtract to: {CONFIG['data_dir']}")
        return
    
    # Prepare data
    print("\n📦 Preparing data...")
    train_loader, val_loader, class_names = prepare_data()
    num_classes = len(class_names)
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Number of classes: {num_classes}")
    
    # Save class names
    class_names_path = os.path.join(BASE_DIR, 'backend', 'models', 'class_names.json')
    with open(class_names_path, 'w', encoding='utf-8') as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    print(f"   Class names saved to: {class_names_path}")
    
    # Build model
    print("\n🏗️ Building model...")
    model = build_model(num_classes)
    
    # Initial training
    print("\n" + "=" * 60)
    print("Phase 1: Feature Extraction (Frozen Base)")
    print("=" * 60)
    history1 = train_model(model, train_loader, val_loader, CONFIG['epochs'], CONFIG['learning_rate'])
    
    # Fine-tuning
    print("\n" + "=" * 60)
    print("Phase 2: Fine-Tuning (Unfrozen Layers)")
    print("=" * 60)
    history2 = fine_tune_model(model, train_loader, val_loader, epochs=10)
    
    # Combine histories
    combined_history = {
        'train_loss': history1['train_loss'] + history2['train_loss'],
        'train_acc': history1['train_acc'] + history2['train_acc'],
        'val_loss': history1['val_loss'] + history2['val_loss'],
        'val_acc': history1['val_acc'] + history2['val_acc'],
    }
    
    # Plot results
    plot_training_history(combined_history, CONFIG['history_save_path'])
    
    # Save training results
    results = {
        'final_train_acc': combined_history['train_acc'][-1],
        'final_val_acc': combined_history['val_acc'][-1],
        'best_val_acc': max(combined_history['val_acc']),
        'total_epochs': len(combined_history['train_acc']),
        'num_classes': num_classes,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'gpu': torch.cuda.get_device_name(0),
    }
    
    with open(CONFIG['results_save_path'], 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print("📊 Training Summary")
    print("=" * 60)
    print(f"   Best Validation Accuracy: {results['best_val_acc']:.2f}%")
    print(f"   Final Training Accuracy:  {results['final_train_acc']:.2f}%")
    print(f"   Final Validation Accuracy: {results['final_val_acc']:.2f}%")
    print(f"   Total Epochs: {results['total_epochs']}")
    print(f"\n✅ Model saved to: {CONFIG['model_save_path']}")
    print(f"✅ Training curves saved to: {CONFIG['history_save_path']}")
    print(f"✅ Results saved to: {CONFIG['results_save_path']}")
    print("\n🎉 Training complete!")

if __name__ == "__main__":
    main()
