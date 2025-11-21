import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import math

# Import our custom modules
from model_swin import SwinTiny
from Dataset.cifar10_reader import CIFAR10Dataset
from utils import check_set_gpu


def create_label_encoder(dataset):
    """Create a mapping from string labels to numeric indices"""
    all_labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        all_labels.append(label)
    
    unique_labels = sorted(list(set(all_labels)))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for idx, label in enumerate(unique_labels)}
    
    return label_to_idx, idx_to_label, unique_labels


def create_swin_model(num_classes, freeze_backbone=True):
    """Create Swin Transformer Tiny model with transfer learning"""
    model = SwinTiny(num_classes=num_classes, pretrained=True)
    
    print(f"Loaded Swin Transformer Tiny pretrained model")
    
    if freeze_backbone:
        # Freeze semua layer kecuali head classifier
        for name, param in model.model.named_parameters():
            if 'head' not in name:  # Freeze semua kecuali head
                param.requires_grad = False
        
        print("Fine-tuning strategy:")
        print("- Frozen: All backbone layers (patch embed, position embed, transformer blocks)")
        print(f"- Trainable: Only classification head ({num_classes} classes)")
        
        # timm 1.0+ sudah handle head dengan benar via num_classes parameter
        # Head sudah berbentuk ClassifierHead dengan global pooling
        print(f"Head structure: {model.model.head}")
    
    return model


def calculate_metrics(y_true, y_pred, num_classes):
    """Calculate accuracy, F1-score, precision, and recall"""
    accuracy = accuracy_score(y_true, y_pred)
    
    # For multiclass classification, use 'weighted' average for imbalanced datasets
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return accuracy, f1, precision, recall


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, label_to_idx, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping and learning rate scheduling"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []

    for batch_idx, batch_data in enumerate(train_loader):
        images, labels, filepath = batch_data
        
        # Labels dari CIFAR-10 sudah berupa integer (0-9), langsung convert ke tensor
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        
        # Pastikan labels adalah 1D tensor [batch_size]
        if labels.dim() > 1:
            labels = labels.squeeze()
            
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Get predictions
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Print progress every 10 batches
        if batch_idx % 10 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    avg_loss = running_loss / len(train_loader)
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_predictions, len(label_to_idx))
    
    return avg_loss, accuracy, f1, precision, recall

def plot_metrics(train_losses, val_losses, train_accs, val_accs, save_dir='.'):
    """Fungsi untuk mem-plot grafik loss dan akurasi."""
    
    # Membuat figure dengan dua subplot (1 baris, 2 kolom)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # --- Plot 1: Loss ---
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Validation Loss', color='orange')
    ax1.set_title('Training vs. Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # --- Plot 2: Accuracy ---
    ax2.plot(train_accs, label='Train Accuracy', color='blue')
    ax2.plot(val_accs, label='Validation Accuracy', color='orange')
    ax2.set_title('Training vs. Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Menampilkan plot
    save_path = os.path.join(save_dir, 'hasil_training_Swin.png')
    plt.savefig(save_path)
    plt.close() 

    print(f"\\nGrafik training telah disimpan sebagai '{save_path}'")

def plot_confusion_matrix(y_true, y_pred, class_names, filename='confusion_matrix_swin.png'):
    """Plot confusion matrix dengan visualisasi yang bagus"""
    # Hitung confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Buat figure
    plt.figure(figsize=(10, 8))
    
    # Plot confusion matrix dengan seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    plt.title('Confusion Matrix - Swin Transformer (CIFAR-10)', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Simpan figure
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix telah disimpan sebagai '{filename}'")

def measure_inference_time(model, val_loader, device, num_warmup=10):
    """Measure inference time metrics"""
    model.eval()
    
    inference_times = []
    total_images = 0
    
    with torch.no_grad():
        # Warmup GPU
        print("Warming up GPU...")
        for i, batch_data in enumerate(val_loader):
            if i >= num_warmup:
                break
            images, _, _ = batch_data
            images = images.to(device)
            _ = model(images)
        
        # Synchronize GPU before timing
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        print("Measuring inference time...")
        start_time = time.time()
        
        # Actual inference timing
        for batch_data in val_loader:
            images, _, _ = batch_data
            images = images.to(device)
            
            # Time each batch
            batch_start = time.time()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            _ = model(images)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_end = time.time()
            
            batch_time = batch_end - batch_start
            inference_times.append(batch_time)
            total_images += images.size(0)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        total_time = time.time() - start_time
    
    # Calculate metrics
    avg_batch_time = np.mean(inference_times)
    std_batch_time = np.std(inference_times)
    avg_time_per_image = total_time / total_images * 1000  # in milliseconds
    throughput = total_images / total_time  # images per second
    
    return {
        'total_time': total_time,
        'total_images': total_images,
        'avg_time_per_image_ms': avg_time_per_image,
        'throughput': throughput,
        'avg_batch_time': avg_batch_time,
        'std_batch_time': std_batch_time
    }

def validate_epoch(model, val_loader, criterion, device, label_to_idx):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in val_loader:
            images, labels, _ = batch_data
            
            # Labels dari CIFAR-10 sudah berupa integer (0-9), langsung convert ke tensor
            if not isinstance(labels, torch.Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            
            # Pastikan labels adalah 1D tensor [batch_size]
            if labels.dim() > 1:
                labels = labels.squeeze()
                
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    accuracy, f1, precision, recall = calculate_metrics(all_labels, all_predictions, len(label_to_idx))
    
    return avg_loss, accuracy, f1, precision, recall, all_labels, all_predictions


def main():
    # Set up device using utils function
    device = check_set_gpu()
    
    # Get script directory for saving outputs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Hyperparameters
    batch_size = 128  # Larger batch size for smaller CIFAR-10 images
    learning_rate = 15e-3 #0.01
    num_epochs = 25  # Fewer epochs since we're using pretrained model
    img_size = 224  # Resize CIFAR-10 (32x32) to 224x224 for Swin Transformer
    
    print(f"Using image size: {img_size}x{img_size}")
    
    # Create datasets - CIFAR-10
    print("Loading CIFAR-10 datasets...")
    data_dir = os.path.join(script_dir, "Dataset", "cifar-10-batches-py")
    
    train_dataset = CIFAR10Dataset(data_dir, split='train', img_size=(img_size, img_size))
    val_dataset = CIFAR10Dataset(data_dir, split='val', img_size=(img_size, img_size))
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create label encoder
    print("Creating label encoder...")
    label_to_idx, idx_to_label, unique_labels = create_label_encoder(train_dataset)
    num_classes = len(unique_labels)
    
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {unique_labels}")
    print(f"Label to index mapping: {label_to_idx}")
    
    cpu_count = os.cpu_count()
    nworkers = cpu_count - 4
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=nworkers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=nworkers)
    
    # Initialize Swin Transformer model with transfer learning
    print("\nInitializing Swin Transformer Tiny model...")
    model = create_swin_model(num_classes, freeze_backbone=True)
    print(model)
    model = model.to(device)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    # Loss function and optimizer setup
    criterion = nn.CrossEntropyLoss()
    
    # Base learning rate untuk Swin Transformer fine-tuning
    base_lr = learning_rate * 0.1
    
    # AdamW optimizer dengan weight decay
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01  # L2 regularization
    )
    
    # Learning rate scheduler setup
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = len(train_loader) * 2  # 2 epochs of warmup
    
    # Learning rate scheduler setup
    num_training_steps = len(train_loader) * num_epochs
    
    # Tentukan hyperparameter untuk scheduler baru
    epochs_per_restart = 10 # Siklus restart pertama akan terjadi setelah 10 epoch

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0 = epochs_per_restart * len(train_loader), # Jumlah langkah untuk siklus pertama
        T_mult = 1, # Faktor pengali durasi setelah restart. 1 = durasi tetap.
        eta_min = 1e-6 # LR minimum yang bisa dicapai
    )

    
    # Gradient clipping
    max_grad_norm = 1.0
    
    print(f"\nStarting training with:")
    print(f"- Device: {device}")
    print(f"- Model: Swin Transformer Tiny (pretrained)")
    print(f"- Image size: 224x224")
    print(f"- Batch size: {batch_size}")
    print(f"- Base learning rate: {base_lr}")
    print(f"- Warmup epochs: 2")
    print(f"- Number of epochs: {num_epochs}")
    print(f"- Weight decay: 0.01")
    print(f"- Gradient clipping: {max_grad_norm}")
    print(f"- Transfer learning: Backbone frozen, head trainable")
    print(f"- Scheduler: Cosine with warmup")
    print("-" * 80)
    
    # Training loop
    best_val_accuracy = 0.0
    best_model_path = os.path.join(script_dir, "best_swin_model.pth")
    
    # Early stopping parameters
    patience = 5
    epochs_no_improve = 0

    # TAMBAHKAN INI: List untuk menyimpan metrik
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-" * 50)
        
        start_time = time.time()
        
        # Training phase
        train_loss, train_acc, train_f1, train_precision, train_recall = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, label_to_idx, max_grad_norm
        )
        
        # Validation phase
        val_loss, val_acc, val_f1, val_precision, val_recall, val_labels, val_preds = validate_epoch(
            model, val_loader, criterion, device, label_to_idx
        )
        
        epoch_time = time.time() - start_time
        
        # Print metrics
        print(f"\nEpoch {epoch+1} Results:")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}, "
              f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}, "
              f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved! Validation accuracy: {val_acc:.4f}")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s)")
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered! No improvement for {patience} epochs.")
            print(f"Best validation accuracy: {best_val_accuracy:.4f}")
            break
        
        print("-" * 80)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    plot_metrics(train_losses, val_losses, train_accs, val_accs, save_dir=script_dir)

    # Load best model for final evaluation
    print("\nLoading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    
    # Final validation evaluation with detailed classification report
    val_loss, val_acc, val_f1, val_precision, val_recall, val_labels, val_preds = validate_epoch(
        model, val_loader, criterion, device, label_to_idx
    )
    
    print("\n" + "="*80)
    print("FINAL EVALUATION RESULTS")
    print("="*80)
    print(f"Final Validation Metrics:")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"F1-Score:  {val_f1:.4f}")
    print(f"Precision: {val_precision:.4f}")
    print(f"Recall:    {val_recall:.4f}")
    
    # Detailed classification report
    print(f"\nDetailed Classification Report:")
    print("-" * 50)
    class_names = sorted(unique_labels)
    print(classification_report(val_labels, val_preds, target_names=[str(cls) for cls in class_names]))
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    confusion_matrix_path = os.path.join(script_dir, 'confusion_matrix_swin.png')
    plot_confusion_matrix(val_labels, val_preds, class_names, filename=confusion_matrix_path)
    
    # Measure inference time
    print("\n" + "="*80)
    print("INFERENCE TIME MEASUREMENT")
    print("="*80)
    inference_metrics = measure_inference_time(model, val_loader, device)
    
    # Get hardware info
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        hardware_info = f"GPU: {gpu_name} ({gpu_memory:.1f} GB)"
    else:
        hardware_info = "CPU"
    
    print(f"\nHardware: {hardware_info}")
    print(f"Total images processed: {inference_metrics['total_images']}")
    print(f"Total inference time: {inference_metrics['total_time']:.3f} seconds")
    print(f"Average time per image: {inference_metrics['avg_time_per_image_ms']:.2f} ms")
    print(f"Throughput: {inference_metrics['throughput']:.2f} images/second")
    print(f"Average batch time: {inference_metrics['avg_batch_time']:.4f} ± {inference_metrics['std_batch_time']:.4f} seconds")
    
    print(f"\nBest model saved as: {best_model_path}")
    
    # Print model summary
    print(f"\nModel Summary:")
    print(f"- Architecture: Swin Transformer Tiny with ImageNet pretraining")
    print(f"- Transfer Learning: Frozen backbone + trainable classifier")
    print(f"- Input size: 224x224x3")  # Swin Tiny input size
    print(f"- Output classes: {num_classes}")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")


if __name__ == "__main__":
    main()

