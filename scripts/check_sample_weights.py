"""
Script to check and visualize sample_weights from the dataset.
This helps verify that sample_weights are being calculated correctly.
"""
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataloader import get_binary_pipelines


def analyze_sample_weights(dataset, num_batches=5, save_dir=None):
    """
    Analyze sample_weights from the dataset.
    
    Args:
        dataset: tf.data.Dataset with (x, y, sample_weights) format
        num_batches: Number of batches to analyze
        save_dir: Directory to save plots (optional)
    """
    all_weights = []
    all_labels = []
    all_roi_ratios = []
    
    print("=" * 60)
    print("Analyzing Sample Weights")
    print("=" * 60)
    
    batch_count = 0
    for batch in dataset.take(num_batches):
        if len(batch) == 3:
            x_batch, y_batch, w_batch = batch
        else:
            print(f"Warning: Batch {batch_count} doesn't have sample_weights!")
            print(f"Batch format: {len(batch)} elements")
            continue
        
        # Convert to numpy for analysis
        weights_np = w_batch.numpy()
        labels_np = y_batch.numpy()
        
        # Calculate ROI intensity ratio (approximate, since we don't have mask here)
        # This is just for reference
        x_np = x_batch.numpy()
        # Assume center region as approximate ROI
        h, w = x_np.shape[1], x_np.shape[2]
        center_h, center_w = h // 2, w // 2
        roi_size = min(h, w) // 3
        roi_region = x_np[:, 
                          center_h - roi_size:center_h + roi_size,
                          center_w - roi_size:center_w + roi_size, :]
        total_intensity = np.sum(x_np, axis=(1, 2, 3))
        roi_intensity = np.sum(roi_region, axis=(1, 2, 3))
        roi_ratio = roi_intensity / (total_intensity + 1e-8)
        
        all_weights.extend(weights_np)
        all_labels.extend(labels_np.flatten())
        all_roi_ratios.extend(roi_ratio)
        
        print(f"\nBatch {batch_count + 1}:")
        print(f"  Shape: x={x_batch.shape}, y={y_batch.shape}, weights={w_batch.shape}")
        print(f"  Weights - Min: {np.min(weights_np):.4f}, Max: {np.max(weights_np):.4f}, "
              f"Mean: {np.mean(weights_np):.4f}, Std: {np.std(weights_np):.4f}")
        print(f"  Labels - Drowsy: {np.sum(labels_np)}, NotDrowsy: {len(labels_np) - np.sum(labels_np)}")
        print(f"  Weight by label - Drowsy mean: {np.mean(weights_np[labels_np.flatten() == 1]):.4f}, "
              f"NotDrowsy mean: {np.mean(weights_np[labels_np.flatten() == 0]):.4f}")
        
        batch_count += 1
    
    if len(all_weights) == 0:
        print("\nNo sample_weights found in dataset!")
        return
    
    # Convert to numpy arrays
    all_weights = np.array(all_weights)
    all_labels = np.array(all_labels)
    all_roi_ratios = np.array(all_roi_ratios)
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("Overall Statistics (across all batches):")
    print("=" * 60)
    print(f"Total samples: {len(all_weights)}")
    print(f"Weights - Min: {np.min(all_weights):.4f}, Max: {np.max(all_weights):.4f}")
    print(f"Weights - Mean: {np.mean(all_weights):.4f}, Median: {np.median(all_weights):.4f}, Std: {np.std(all_weights):.4f}")
    print(f"Weights - 25th percentile: {np.percentile(all_weights, 25):.4f}")
    print(f"Weights - 75th percentile: {np.percentile(all_weights, 75):.4f}")
    print(f"\nBy Label:")
    print(f"  Drowsy samples: {np.sum(all_labels == 1)} (mean weight: {np.mean(all_weights[all_labels == 1]):.4f})")
    print(f"  NotDrowsy samples: {np.sum(all_labels == 0)} (mean weight: {np.mean(all_weights[all_labels == 0]):.4f})")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of weights
    axes[0, 0].hist(all_weights, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(all_weights), color='red', linestyle='--', label=f'Mean: {np.mean(all_weights):.4f}')
    axes[0, 0].axvline(np.median(all_weights), color='green', linestyle='--', label=f'Median: {np.median(all_weights):.4f}')
    axes[0, 0].set_xlabel('Sample Weight')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Sample Weights')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Weights by label
    drowsy_weights = all_weights[all_labels == 1]
    notdrowsy_weights = all_weights[all_labels == 0]
    axes[0, 1].hist([notdrowsy_weights, drowsy_weights], bins=30, 
                    label=['NotDrowsy', 'Drowsy'], alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel('Sample Weight')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Sample Weights by Class')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Box plot by label
    box_data = [notdrowsy_weights, drowsy_weights]
    axes[1, 0].boxplot(box_data, labels=['NotDrowsy', 'Drowsy'])
    axes[1, 0].set_ylabel('Sample Weight')
    axes[1, 0].set_title('Sample Weights Distribution by Class')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Scatter: ROI ratio vs Weight
    axes[1, 1].scatter(all_roi_ratios, all_weights, alpha=0.5, s=20, c=all_labels, cmap='coolwarm')
    axes[1, 1].set_xlabel('ROI Intensity Ratio (approximate)')
    axes[1, 1].set_ylabel('Sample Weight')
    axes[1, 1].set_title('ROI Ratio vs Sample Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "sample_weights_analysis.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    return {
        'weights': all_weights,
        'labels': all_labels,
        'stats': {
            'mean': np.mean(all_weights),
            'std': np.std(all_weights),
            'min': np.min(all_weights),
            'max': np.max(all_weights),
            'median': np.median(all_weights)
        }
    }


def check_dataset_format(dataset, num_batches=3):
    """
    Check if dataset has sample_weights and show format.
    """
    print("=" * 60)
    print("Checking Dataset Format")
    print("=" * 60)
    
    for i, batch in enumerate(dataset.take(num_batches)):
        print(f"\nBatch {i + 1}:")
        print(f"  Type: {type(batch)}")
        if isinstance(batch, (tuple, list)):
            print(f"  Number of elements: {len(batch)}")
            for j, elem in enumerate(batch):
                if hasattr(elem, 'shape'):
                    print(f"    Element {j}: shape={elem.shape}, dtype={elem.dtype}")
                else:
                    print(f"    Element {j}: type={type(elem)}")
        else:
            print(f"  Shape: {batch.shape if hasattr(batch, 'shape') else 'N/A'}")


if __name__ == "__main__":
    # Project root
    project_root = Path(__file__).parent.parent
    output_dir = os.path.join(project_root, "splitted_dataset")
    
    if not os.path.exists(output_dir):
        print(f"Error: Dataset directory not found: {output_dir}")
        print("Please make sure the dataset is split first.")
        sys.exit(1)
    
    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = get_binary_pipelines(
        output_dir,
        img_size=(224, 224),
        batch_size=16,
        seed=42,
        use_masks=True  # Enable masks to get sample_weights
    )
    
    print("\n" + "=" * 60)
    print("TRAINING DATASET")
    print("=" * 60)
    check_dataset_format(train_ds, num_batches=2)
    
    print("\n" + "=" * 60)
    print("Analyzing Training Dataset Sample Weights")
    print("=" * 60)
    results = analyze_sample_weights(
        train_ds, 
        num_batches=10,
        save_dir=os.path.join(project_root, "sample_weights_analysis")
    )
    
    print("\n" + "=" * 60)
    print("VALIDATION DATASET")
    print("=" * 60)
    check_dataset_format(val_ds, num_batches=2)
    
    # Note: val_ds might not have sample_weights if use_masks is only applied to train
    if len(next(iter(val_ds))) == 3:
        print("\nAnalyzing Validation Dataset Sample Weights...")
        analyze_sample_weights(val_ds, num_batches=5)
    else:
        print("\nValidation dataset doesn't have sample_weights (expected if use_masks only for train)")
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

