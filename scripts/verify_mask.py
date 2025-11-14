"""
Script to verify that the mask is working correctly.
Checks if sample weights make sense based on ROI (eye-mouth) regions.
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
from src.simple_mask import create_simple_mask_generator


def calculate_mask_coverage(img_size=(224, 224)):
    """Calculate what percentage of the image is covered by the mask."""
    h, w = img_size
    
    # Eye region
    eye_top = int(0.2 * h)
    eye_bottom = int(0.53 * h)
    eye_left = int(0.1 * w)
    eye_right = int(0.9 * w)
    eye_area = (eye_bottom - eye_top) * (eye_right - eye_left)
    
    # Mouth region
    mouth_top = int(0.57 * h)
    mouth_bottom = int(0.9 * h)
    mouth_left = int(0.2 * w)
    mouth_right = int(0.8 * w)
    mouth_area = (mouth_bottom - mouth_top) * (mouth_right - mouth_left)
    
    # Check for overlap
    overlap_top = max(eye_bottom, mouth_top)
    overlap_bottom = min(eye_bottom, mouth_bottom)
    if overlap_top < overlap_bottom:
        # There's vertical overlap, check horizontal
        overlap_left = max(eye_left, mouth_left)
        overlap_right = min(eye_right, mouth_right)
        if overlap_left < overlap_right:
            overlap_area = (overlap_bottom - overlap_top) * (overlap_right - overlap_left)
        else:
            overlap_area = 0
    else:
        overlap_area = 0
    
    total_roi_area = eye_area + mouth_area - overlap_area
    total_image_area = h * w
    coverage = total_roi_area / total_image_area
    
    return {
        'eye_area': eye_area,
        'mouth_area': mouth_area,
        'overlap_area': overlap_area,
        'total_roi_area': total_roi_area,
        'total_image_area': total_image_area,
        'coverage_percent': coverage * 100
    }


def analyze_mask_effectiveness(dataset, num_batches=10):
    """
    Analyze if the mask is working correctly by checking:
    1. ROI intensity ratio (should be reasonable based on mask coverage)
    2. Sample weight distribution
    3. Correlation between ROI intensity and sample weights
    """
    mask_generator = create_simple_mask_generator(img_size=(224, 224), use_soft_mask=False)
    
    all_focus_ratios = []
    all_sample_weights = []
    all_roi_intensities = []
    all_total_intensities = []
    all_labels = []
    
    print("=" * 70)
    print("Mask Effectiveness Analysis")
    print("=" * 70)
    
    # Calculate expected mask coverage
    coverage_info = calculate_mask_coverage()
    print(f"\nMask Coverage Information:")
    print(f"  Eye region: {coverage_info['eye_area']} pixels")
    print(f"  Mouth region: {coverage_info['mouth_area']} pixels")
    print(f"  Overlap: {coverage_info['overlap_area']} pixels")
    print(f"  Total ROI area: {coverage_info['total_roi_area']} pixels")
    print(f"  Total image area: {coverage_info['total_image_area']} pixels")
    print(f"  ROI coverage: {coverage_info['coverage_percent']:.2f}% of image")
    
    print(f"\nExpected focus_ratio range:")
    print(f"  If ROI has same intensity as background: ~{coverage_info['coverage_percent']/100:.3f}")
    print(f"  If ROI is brighter (typical): ~0.4-0.7")
    print(f"  If ROI is much brighter: >0.7")
    
    batch_count = 0
    for batch in dataset.take(num_batches):
        if len(batch) != 3:
            print(f"Warning: Batch {batch_count + 1} doesn't have sample_weights!")
            continue
        
        x_batch, y_batch, w_batch = batch
        
        # Generate masks
        masks = mask_generator.generate_mask(x_batch)
        
        # Calculate intensities
        roi_intensity = tf.reduce_sum(x_batch * masks, axis=[1, 2, 3]).numpy()
        total_intensity = tf.reduce_sum(x_batch, axis=[1, 2, 3]).numpy()
        focus_ratio = roi_intensity / (total_intensity + 1e-8)
        
        weights_np = w_batch.numpy()
        labels_np = y_batch.numpy().flatten()
        
        all_focus_ratios.extend(focus_ratio)
        all_sample_weights.extend(weights_np)
        all_roi_intensities.extend(roi_intensity)
        all_total_intensities.extend(total_intensity)
        all_labels.extend(labels_np)
        
        batch_count += 1
    
    if len(all_focus_ratios) == 0:
        print("\nNo data collected!")
        return
    
    # Convert to numpy
    all_focus_ratios = np.array(all_focus_ratios)
    all_sample_weights = np.array(all_sample_weights)
    all_roi_intensities = np.array(all_roi_intensities)
    all_total_intensities = np.array(all_total_intensities)
    all_labels = np.array(all_labels)
    
    # Statistics
    print("\n" + "=" * 70)
    print("Focus Ratio Statistics (ROI intensity / Total intensity):")
    print("=" * 70)
    print(f"  Mean: {np.mean(all_focus_ratios):.4f}")
    print(f"  Median: {np.median(all_focus_ratios):.4f}")
    print(f"  Std: {np.std(all_focus_ratios):.4f}")
    print(f"  Min: {np.min(all_focus_ratios):.4f}, Max: {np.max(all_focus_ratios):.4f}")
    print(f"  25th percentile: {np.percentile(all_focus_ratios, 25):.4f}")
    print(f"  75th percentile: {np.percentile(all_focus_ratios, 75):.4f}")
    
    print("\n" + "=" * 70)
    print("Sample Weight Statistics (after normalization):")
    print("=" * 70)
    print(f"  Mean: {np.mean(all_sample_weights):.4f} (should be ~1.0)")
    print(f"  Median: {np.median(all_sample_weights):.4f}")
    print(f"  Std: {np.std(all_sample_weights):.4f}")
    print(f"  Min: {np.min(all_sample_weights):.4f}, Max: {np.max(all_sample_weights):.4f}")
    
    # Correlation analysis
    correlation = np.corrcoef(all_focus_ratios, all_sample_weights)[0, 1]
    print("\n" + "=" * 70)
    print("Correlation Analysis:")
    print("=" * 70)
    print(f"  Focus ratio vs Sample weight correlation: {correlation:.4f}")
    print(f"  (Should be positive and high, ideally >0.8)")
    
    # Check if focus_ratio makes sense
    expected_min = coverage_info['coverage_percent'] / 100 * 0.5  # At least 50% of coverage
    expected_max = coverage_info['coverage_percent'] / 100 * 2.0  # At most 200% of coverage
    
    print("\n" + "=" * 70)
    print("Mask Validation:")
    print("=" * 70)
    
    if np.mean(all_focus_ratios) < expected_min:
        print(f"  ⚠️  WARNING: Mean focus_ratio ({np.mean(all_focus_ratios):.4f}) is very low!")
        print(f"     Expected at least {expected_min:.4f} based on mask coverage.")
        print(f"     This suggests ROI might not be capturing important regions.")
    elif np.mean(all_focus_ratios) > expected_max:
        print(f"  ⚠️  WARNING: Mean focus_ratio ({np.mean(all_focus_ratios):.4f}) is very high!")
        print(f"     Expected at most {expected_max:.4f} based on mask coverage.")
        print(f"     This might indicate an issue with mask calculation.")
    else:
        print(f"  ✅ Mean focus_ratio ({np.mean(all_focus_ratios):.4f}) is within reasonable range.")
    
    if correlation < 0.5:
        print(f"  ⚠️  WARNING: Low correlation ({correlation:.4f}) between focus_ratio and sample_weight!")
        print(f"     This suggests sample_weight calculation might not be working correctly.")
    elif correlation > 0.8:
        print(f"  ✅ High correlation ({correlation:.4f}) - sample_weight correctly reflects ROI intensity.")
    else:
        print(f"  ⚠️  Moderate correlation ({correlation:.4f}) - sample_weight partially reflects ROI intensity.")
    
    if abs(np.mean(all_sample_weights) - 1.0) > 0.01:
        print(f"  ⚠️  WARNING: Sample weight mean ({np.mean(all_sample_weights):.4f}) is not ~1.0!")
        print(f"     Normalization might not be working correctly.")
    else:
        print(f"  ✅ Sample weight mean is correctly normalized to ~1.0.")
    
    # By class analysis
    drowsy_focus = all_focus_ratios[all_labels == 1]
    notdrowsy_focus = all_focus_ratios[all_labels == 0]
    
    print("\n" + "=" * 70)
    print("Focus Ratio by Class:")
    print("=" * 70)
    print(f"  Drowsy: mean={np.mean(drowsy_focus):.4f}, std={np.std(drowsy_focus):.4f}")
    print(f"  NotDrowsy: mean={np.mean(notdrowsy_focus):.4f}, std={np.std(notdrowsy_focus):.4f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Focus ratio distribution
    axes[0, 0].hist(all_focus_ratios, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(all_focus_ratios), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_focus_ratios):.4f}')
    axes[0, 0].axvline(coverage_info['coverage_percent']/100, color='green', linestyle='--', 
                       label=f'Expected (area%): {coverage_info["coverage_percent"]/100:.3f}')
    axes[0, 0].set_xlabel('Focus Ratio (ROI intensity / Total intensity)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Focus Ratio Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Focus ratio vs Sample weight
    axes[0, 1].scatter(all_focus_ratios, all_sample_weights, alpha=0.5, s=20, c=all_labels, cmap='coolwarm')
    axes[0, 1].set_xlabel('Focus Ratio')
    axes[0, 1].set_ylabel('Sample Weight')
    axes[0, 1].set_title(f'Focus Ratio vs Sample Weight (corr={correlation:.3f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Focus ratio by class
    axes[1, 0].hist([notdrowsy_focus, drowsy_focus], bins=30, 
                    label=['NotDrowsy', 'Drowsy'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Focus Ratio')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Focus Ratio by Class')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Sample weight distribution
    axes[1, 1].hist(all_sample_weights, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(np.mean(all_sample_weights), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_sample_weights):.4f}')
    axes[1, 1].set_xlabel('Sample Weight')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Sample Weight Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(project_root, "sample_weights_analysis", "mask_verification.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.close()
    
    return {
        'focus_ratios': all_focus_ratios,
        'sample_weights': all_sample_weights,
        'correlation': correlation,
        'coverage_info': coverage_info
    }


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    output_dir = os.path.join(project_root, "splitted_dataset")
    
    if not os.path.exists(output_dir):
        print(f"Error: Dataset directory not found: {output_dir}")
        sys.exit(1)
    
    print("Loading datasets...")
    train_ds, val_ds, test_ds, class_names = get_binary_pipelines(
        output_dir,
        img_size=(224, 224),
        batch_size=16,
        seed=42,
        use_masks=True
    )
    
    print("\nAnalyzing mask effectiveness...")
    results = analyze_mask_effectiveness(train_ds, num_batches=10)
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

