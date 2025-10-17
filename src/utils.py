import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
import os

def create_run_directories(run_name="run_001"):
    """Create basic run directory structure"""
    base_dir = "runs"
    run_dir = os.path.join(base_dir, run_name)
    
    # Create directories
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "plots"), exist_ok=True)
    
    print(f"📁 Created run directories: {run_dir}")
    return run_dir

def plot_dataset_distribution(data_dir="data", save_path=None):
    """Plot distribution of drowsy vs notdrowsy in train/val/test datasets"""
    import glob
    
    datasets = ['train', 'val', 'test']
    drowsy_counts = []
    notdrowsy_counts = []
    
    for dataset in datasets:
        dataset_path = os.path.join(data_dir, dataset)
        if not os.path.exists(dataset_path):
            drowsy_counts.append(0)
            notdrowsy_counts.append(0)
            continue

        # Support two layouts:
        # 1) Old: files named with *_1.* (drowsy) and *_0.* (not drowsy)
        # 2) New: class subfolders {NotDrowsy, Drowsy} under each split
        old_style_drowsy = glob.glob(os.path.join(dataset_path, "*_1.*"))
        old_style_notdrowsy = glob.glob(os.path.join(dataset_path, "*_0.*"))

        notdrowsy_dir = os.path.join(dataset_path, 'NotDrowsy')
        drowsy_dir = os.path.join(dataset_path, 'Drowsy')

        if (len(old_style_drowsy) + len(old_style_notdrowsy)) > 0:
            drowsy_counts.append(len(old_style_drowsy))
            notdrowsy_counts.append(len(old_style_notdrowsy))
        elif os.path.isdir(notdrowsy_dir) and os.path.isdir(drowsy_dir):
            nd_count = sum(
                1 for f in os.listdir(notdrowsy_dir)
                if os.path.isfile(os.path.join(notdrowsy_dir, f))
            )
            d_count = sum(
                1 for f in os.listdir(drowsy_dir)
                if os.path.isfile(os.path.join(drowsy_dir, f))
            )
            notdrowsy_counts.append(nd_count)
            drowsy_counts.append(d_count)
        else:
            # Unknown layout; count as zero to avoid crashes
            drowsy_counts.append(0)
            notdrowsy_counts.append(0)
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    x = np.arange(len(datasets))
    width = 0.35
    
    ax1.bar(x - width/2, drowsy_counts, width, label='Drowsy', color='red')
    ax1.bar(x + width/2, notdrowsy_counts, width, label='Not Drowsy', color='blue')
    
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Dataset Distribution: Drowsy vs Not Drowsy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True)
    
    # Add value labels on bars
    for i, (d, nd) in enumerate(zip(drowsy_counts, notdrowsy_counts)):
        ax1.text(i - width/2, d + max(drowsy_counts + notdrowsy_counts) * 0.01, str(d), 
                ha='center', va='bottom', fontweight='bold')
        ax1.text(i + width/2, nd + max(drowsy_counts + notdrowsy_counts) * 0.01, str(nd), 
                ha='center', va='bottom', fontweight='bold')
    
    # Pie chart for total distribution
    total_drowsy = sum(drowsy_counts)
    total_notdrowsy = sum(notdrowsy_counts)
    
    if (total_drowsy + total_notdrowsy) > 0:
        ax2.pie([total_drowsy, total_notdrowsy], 
                labels=[f'Drowsy ({total_drowsy})', f'Not Drowsy ({total_notdrowsy})'],
                autopct='%1.1f%%', startangle=90, colors=['red', 'blue'])
        ax2.set_title('Total Dataset Distribution')
    else:
        ax2.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=12)
        ax2.set_title('Total Dataset Distribution')
        ax2.axis('off')
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Dataset distribution plot saved: {save_path}")
    
    plt.show()
    
    # Print summary
    print("\n📊 Dataset Distribution Summary:")
    print("=" * 40)
    for i, dataset in enumerate(datasets):
        total = drowsy_counts[i] + notdrowsy_counts[i]
        if total > 0:
            drowsy_pct = (drowsy_counts[i] / total) * 100
            notdrowsy_pct = (notdrowsy_counts[i] / total) * 100
            print(f"{dataset.capitalize():>8}: {drowsy_counts[i]:>4} drowsy ({drowsy_pct:>5.1f}%) | {notdrowsy_counts[i]:>4} not drowsy ({notdrowsy_pct:>5.1f}%) | Total: {total}")
        else:
            print(f"{dataset.capitalize():>8}: No data found")
    
    print(f"\nTotal Images: {total_drowsy + total_notdrowsy}")
    print(f"Overall Drowsy Ratio: {(total_drowsy / (total_drowsy + total_notdrowsy)) * 100:.1f}%")

def plot_history(history, save_path=None):
    """Plot training history: accuracy and loss"""
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    # Check for different accuracy metric names
    acc_key = None
    val_acc_key = None
    
    # Look for accuracy metrics (could be 'accuracy', 'weighted_accuracy', etc.)
    for key in history.history.keys():
        if key == 'accuracy' or key.endswith('_accuracy'):
            acc_key = key
        elif key == 'val_accuracy' or key.endswith('_val_accuracy'):
            val_acc_key = key
    
    if acc_key:
        plt.plot(history.history[acc_key], label='Train')
    if val_acc_key:
        plt.plot(history.history[val_acc_key], label='Validation')
    
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved: {save_path}")
    
    plt.show()

def plot_metrics(history, save_path=None):
    """Plot additional metrics: precision, recall, auc"""
    # Look for metrics with different naming patterns
    metrics = ['precision', 'recall', 'auc']
    available_metrics = []
    
    # Check for different metric naming patterns
    for metric in metrics:
        # Look for exact match or weighted version
        for key in history.history.keys():
            if key == metric or key.endswith(f'_{metric}') or key.startswith(f'{metric}_'):
                available_metrics.append((metric, key))
                break
    
    if not available_metrics:
        print("No additional metrics found in history")
        print(f"Available keys: {list(history.history.keys())}")
        return
    
    plt.figure(figsize=(15, 5))
    for i, (metric, key) in enumerate(available_metrics):
        plt.subplot(1, len(available_metrics), i+1)
        plt.plot(history.history[key], label=f'Train {metric}')
        
        # Look for validation version
        val_key = None
        for val_key_candidate in history.history.keys():
            if val_key_candidate == f'val_{metric}' or val_key_candidate.endswith(f'_val_{metric}') or val_key_candidate.startswith(f'val_{metric}_'):
                val_key = val_key_candidate
                break
        
        if val_key:
            plt.plot(history.history[val_key], label=f'Val {metric}')
        
        plt.title(f'{metric.upper()}')
        plt.xlabel('Epoch')
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved: {save_path}")
    
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=['Not Drowsy', 'Drowsy'], save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved: {save_path}")
    
    plt.show()

def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved: {save_path}")
    
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    # Save plot if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Plot saved: {save_path}")
    
    plt.show()