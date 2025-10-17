"""
Test script to visualize masks and check if they're working correctly.
"""
import numpy as np
import matplotlib.pyplot as plt
import os

def test_mask_generation():
    """Test mask generation and visualize results."""
    print("🧪 Testing mask generation...")
    
    # Import after checking if we're in the right environment
    try:
        from src.simple_mask import create_simple_mask_generator
        import tensorflow as tf
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're in the correct environment with TensorFlow installed.")
        return False
    
    # Create mask generator
    mask_gen = create_simple_mask_generator(img_size=(224, 224))
    
    # Create dummy batch
    batch_images = tf.random.normal((2, 224, 224, 3))
    
    # Generate masks
    masks = mask_gen.generate_mask(batch_images)
    
    print(f"✅ Mask shape: {masks.shape}")
    print(f"✅ Mask value range: [{tf.reduce_min(masks):.3f}, {tf.reduce_max(masks):.3f}]")
    
    # Convert to numpy for visualization
    mask_np = masks[0].numpy()  # First sample
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Original mask
    plt.subplot(1, 3, 1)
    plt.imshow(mask_np[:, :, 0], cmap='hot')
    plt.title('Generated Mask')
    plt.colorbar()
    
    # Expected regions overlay
    plt.subplot(1, 3, 2)
    plt.imshow(mask_np[:, :, 0], cmap='hot')
    
    # Draw expected eye region
    eye_top, eye_bottom = int(0.25 * 224), int(0.45 * 224)
    eye_left, eye_right = int(0.2 * 224), int(0.8 * 224)
    plt.axhline(y=eye_top, color='blue', linestyle='--', alpha=0.7, label='Eye region')
    plt.axhline(y=eye_bottom, color='blue', linestyle='--', alpha=0.7)
    plt.axvline(x=eye_left, color='blue', linestyle='--', alpha=0.7)
    plt.axvline(x=eye_right, color='blue', linestyle='--', alpha=0.7)
    
    # Draw expected mouth region
    mouth_top, mouth_bottom = int(0.55 * 224), int(0.75 * 224)
    mouth_left, mouth_right = int(0.3 * 224), int(0.7 * 224)
    plt.axhline(y=mouth_top, color='green', linestyle='--', alpha=0.7, label='Mouth region')
    plt.axhline(y=mouth_bottom, color='green', linestyle='--', alpha=0.7)
    plt.axvline(x=mouth_left, color='green', linestyle='--', alpha=0.7)
    plt.axvline(x=mouth_right, color='green', linestyle='--', alpha=0.7)
    
    plt.title('Mask with Expected Regions')
    plt.legend()
    plt.colorbar()
    
    # Sample weight calculation
    plt.subplot(1, 3, 3)
    sample_weight = tf.reduce_mean(masks, axis=[1, 2, 3])
    plt.bar(['Sample 1', 'Sample 2'], sample_weight.numpy())
    plt.title('Sample Weights')
    plt.ylabel('Weight Value')
    
    plt.tight_layout()
    plt.savefig('mask_test_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✅ Mask test completed! Check mask_test_visualization.png")
    
    # Check if mask values are reasonable
    eye_region = mask_np[eye_top:eye_bottom, eye_left:eye_right, 0]
    mouth_region = mask_np[mouth_top:mouth_bottom, mouth_left:mouth_right, 0]
    other_region = np.concatenate([
        mask_np[:eye_top, :, 0].flatten(),
        mask_np[eye_bottom:mouth_top, :, 0].flatten(),
        mask_np[mouth_bottom:, :, 0].flatten()
    ])
    
    print(f"\n📊 Mask Analysis:")
    print(f"Eye region mean: {np.mean(eye_region):.3f} (should be ~1.0)")
    print(f"Mouth region mean: {np.mean(mouth_region):.3f} (should be ~1.0)")
    print(f"Other regions mean: {np.mean(other_region):.3f} (should be ~0.0)")
    print(f"Sample weights: {sample_weight.numpy()}")
    
    return True

if __name__ == "__main__":
    test_mask_generation()
