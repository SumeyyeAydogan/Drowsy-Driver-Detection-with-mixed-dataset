"""
Analyze sample_weights_stats.json to understand how sample weights evolve during training.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_sample_weights_stats(json_path):
    """Load and analyze sample weights statistics."""
    with open(json_path, 'r') as f:
        stats = json.load(f)
    
    epochs = [s['epoch'] for s in stats]
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    mins = [s['min'] for s in stats]
    maxs = [s['max'] for s in stats]
    medians = [s['median'] for s in stats]
    drowsy_means = [s['drowsy_mean'] for s in stats]
    notdrowsy_means = [s['notdrowsy_mean'] for s in stats]
    
    print("=" * 70)
    print("Sample Weights Statistics Analysis (30 Epochs)")
    print("=" * 70)
    
    # Overall statistics
    print("\nOverall Statistics Across All Epochs:")
    print(f"  Mean: {np.mean(means):.6f} (should be ~1.0)")
    print(f"  Std range: [{np.min(stds):.4f}, {np.max(stds):.4f}]")
    print(f"  Min range: [{np.min(mins):.4f}, {np.max(mins):.4f}]")
    print(f"  Max range: [{np.min(maxs):.4f}, {np.max(maxs):.4f}]")
    print(f"  Median range: [{np.min(medians):.4f}, {np.max(medians):.4f}]")
    
    # Class differences
    drowsy_avg = np.mean(drowsy_means)
    notdrowsy_avg = np.mean(notdrowsy_means)
    class_diff = drowsy_avg - notdrowsy_avg
    
    print("\nClass Differences:")
    print(f"  Drowsy average mean: {drowsy_avg:.4f}")
    print(f"  NotDrowsy average mean: {notdrowsy_avg:.4f}")
    print(f"  Average difference: {class_diff:.4f} ({abs(class_diff)*100:.2f}%)")
    
    # Find epochs with extreme values
    max_std_epoch = epochs[np.argmax(stds)]
    min_std_epoch = epochs[np.argmin(stds)]
    max_diff_epoch = epochs[np.argmax([abs(d - n) for d, n in zip(drowsy_means, notdrowsy_means)])]
    
    print("\nExtreme Values:")
    print(f"  Highest std: Epoch {max_std_epoch} (std={max(stds):.4f})")
    print(f"  Lowest std: Epoch {min_std_epoch} (std={min(stds):.4f})")
    print(f"  Largest class difference: Epoch {max_diff_epoch}")
    print(f"    Drowsy: {drowsy_means[max_diff_epoch-1]:.4f}, NotDrowsy: {notdrowsy_means[max_diff_epoch-1]:.4f}")
    
    # Stability check
    std_of_stds = np.std(stds)
    std_of_means = np.std(means)
    
    print("\nStability Analysis:")
    print(f"  Std of stds: {std_of_stds:.6f} (lower is more stable)")
    print(f"  Std of means: {std_of_means:.6f} (should be very low, <0.001)")
    
    if std_of_means < 0.001:
        print("  ✅ Mean is very stable across epochs (normalization working correctly)")
    else:
        print("  ⚠️  Mean varies across epochs (might indicate normalization issues)")
    
    if std_of_stds < 0.01:
        print("  ✅ Std is very stable across epochs (consistent variance)")
    else:
        print("  ⚠️  Std varies significantly across epochs")
    
    # Trend analysis
    std_trend = np.polyfit(epochs, stds, 1)[0]
    drowsy_trend = np.polyfit(epochs, drowsy_means, 1)[0]
    notdrowsy_trend = np.polyfit(epochs, notdrowsy_means, 1)[0]
    
    print("\nTrend Analysis (linear regression slope):")
    print(f"  Std trend: {std_trend:.6f} per epoch (positive = increasing, negative = decreasing)")
    print(f"  Drowsy mean trend: {drowsy_trend:.6f} per epoch")
    print(f"  NotDrowsy mean trend: {notdrowsy_trend:.6f} per epoch")
    
    if abs(std_trend) < 0.0001:
        print("  ✅ Std is stable (no significant trend)")
    elif std_trend > 0:
        print("  ⚠️  Std is increasing (variance growing over time)")
    else:
        print("  ⚠️  Std is decreasing (variance shrinking over time)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Mean over epochs (should be ~1.0)
    axes[0, 0].plot(epochs, means, 'b-', linewidth=2, marker='o', markersize=4)
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Target (1.0)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Sample Weight')
    axes[0, 0].set_title('Mean Sample Weight Over Epochs')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0.999, 1.001])
    
    # 2. Std over epochs
    axes[0, 1].plot(epochs, stds, 'g-', linewidth=2, marker='o', markersize=4)
    axes[0, 1].axhline(y=np.mean(stds), color='r', linestyle='--', 
                        label=f'Average: {np.mean(stds):.4f}')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Std')
    axes[0, 1].set_title('Standard Deviation Over Epochs')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Min/Max over epochs
    axes[0, 2].plot(epochs, mins, 'b-', linewidth=1.5, marker='o', markersize=3, label='Min')
    axes[0, 2].plot(epochs, maxs, 'r-', linewidth=1.5, marker='s', markersize=3, label='Max')
    axes[0, 2].fill_between(epochs, mins, maxs, alpha=0.2, color='gray')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Sample Weight')
    axes[0, 2].set_title('Min/Max Range Over Epochs')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Class means over epochs
    axes[1, 0].plot(epochs, drowsy_means, 'r-', linewidth=2, marker='o', markersize=4, label='Drowsy')
    axes[1, 0].plot(epochs, notdrowsy_means, 'b-', linewidth=2, marker='s', markersize=4, label='NotDrowsy')
    axes[1, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean Sample Weight')
    axes[1, 0].set_title('Mean Sample Weight by Class Over Epochs')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Class difference over epochs
    class_diffs = [d - n for d, n in zip(drowsy_means, notdrowsy_means)]
    axes[1, 1].plot(epochs, class_diffs, 'purple', linewidth=2, marker='o', markersize=4)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=np.mean(class_diffs), color='r', linestyle='--', 
                        label=f'Average: {np.mean(class_diffs):.4f}')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Difference (Drowsy - NotDrowsy)')
    axes[1, 1].set_title('Class Difference Over Epochs')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Distribution of stds
    axes[1, 2].hist(stds, bins=15, edgecolor='black', alpha=0.7, color='green')
    axes[1, 2].axvline(np.mean(stds), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(stds):.4f}')
    axes[1, 2].set_xlabel('Std')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Distribution of Std Values')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = Path(json_path).parent / "sample_weights_evolution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()
    
    return {
        'epochs': epochs,
        'means': means,
        'stds': stds,
        'drowsy_means': drowsy_means,
        'notdrowsy_means': notdrowsy_means
    }


if __name__ == "__main__":
    import sys
    json_path = "sample_weights_stats.json"
    
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    analyze_sample_weights_stats(json_path)

