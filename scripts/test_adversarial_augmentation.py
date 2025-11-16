"""
Test script for adversarial background augmentation.
Visualizes how augmentation affects images before and after.
"""

import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adversarial_augmentation import RandomBackgroundReplacement, BackgroundAugmentation


def test_single_image(image_path, output_dir="test_outputs", prob=1.0, face_ratio=0.4):
    """
    Test adversarial augmentation on a single image.
    
    Args:
        image_path: Path to test image
        output_dir: Directory to save output images
        prob: Probability of applying augmentation (1.0 = always apply for testing)
        face_ratio: Face region ratio
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print(f"📷 Loading image: {image_path}")
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.utils.img_to_array(image) / 255.0
    image_batch = tf.expand_dims(image_array, 0)  # Add batch dimension
    
    print(f"✅ Image loaded: {image_array.shape}")
    
    # Test RandomBackgroundReplacement (simple version)
    print("\n🔄 Testing RandomBackgroundReplacement...")
    bg_aug_simple = RandomBackgroundReplacement(prob=prob, face_center_ratio=face_ratio)
    augmented_simple = bg_aug_simple(image_batch, training=True)
    
    # Test BackgroundAugmentation (advanced version)
    print("🔄 Testing BackgroundAugmentation (advanced)...")
    bg_aug_advanced = BackgroundAugmentation(
        background_replace_prob=0.3,
        background_blur_prob=0.2,
        background_noise_prob=0.2
    )
    augmented_advanced = bg_aug_advanced(image_batch, training=True)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original
    axes[0].imshow(image_array)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Simple augmentation
    axes[1].imshow(augmented_simple[0].numpy())
    axes[1].set_title("RandomBackgroundReplacement\n(Simple Version)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Advanced augmentation
    axes[2].imshow(augmented_advanced[0].numpy())
    axes[2].set_title("BackgroundAugmentation\n(Advanced Version)", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, "adversarial_aug_test.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    
    plt.show()
    
    return augmented_simple, augmented_advanced


def test_multiple_augmentations(image_path, output_dir="test_outputs", num_samples=6, face_ratio=0.4):
    """
    Test multiple random augmentations on the same image to show variety.
    
    Args:
        image_path: Path to test image
        output_dir: Directory to save output images
        num_samples: Number of different augmentations to generate
        face_ratio: Face region ratio
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print(f"📷 Loading image: {image_path}")
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return
    
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.utils.img_to_array(image) / 255.0
    image_batch = tf.expand_dims(image_array, 0)
    
    # Create augmentation layer
    bg_aug = RandomBackgroundReplacement(prob=1.0, face_center_ratio=face_ratio)
    
    # Generate multiple augmented versions
    print(f"\n🔄 Generating {num_samples} different augmentations...")
    augmented_images = []
    for i in range(num_samples):
        augmented = bg_aug(image_batch, training=True)
        augmented_images.append(augmented[0].numpy())
        print(f"  ✅ Generated augmentation {i+1}/{num_samples}")
    
    # Create grid visualization
    cols = 3
    rows = (num_samples + 2) // cols  # +2 for original and one more row
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Original in first position
    axes[0, 0].imshow(image_array)
    axes[0, 0].set_title("Original", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Augmented versions
    for i, aug_img in enumerate(augmented_images):
        row = (i + 1) // cols
        col = (i + 1) % cols
        if row < rows and col < cols:
            axes[row, col].imshow(aug_img)
            axes[row, col].set_title(f"Augmented #{i+1}", fontsize=12)
            axes[row, col].axis('off')
    
    # Hide unused subplots
    for i in range(len(augmented_images) + 1, rows * cols):
        row = i // cols
        col = i % cols
        if row < rows and col < cols:
            axes[row, col].axis('off')
    
    plt.suptitle("Adversarial Background Augmentation - Multiple Samples", 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    output_path = os.path.join(output_dir, "adversarial_aug_multiple.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    
    plt.show()


def test_batch_augmentation(dataset_dir, output_dir="test_outputs", num_images=4, prob=0.5, face_ratio=0.4):
    """
    Test augmentation on a batch of images from dataset.
    
    Args:
        dataset_dir: Directory containing test images (or path to single image)
        output_dir: Directory to save output images
        num_images: Number of images to test
        prob: Probability of applying augmentation
        face_ratio: Face region ratio
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    if os.path.isfile(dataset_dir):
        # Single image
        image_paths = [dataset_dir]
    else:
        # Directory - find image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_paths = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
                    if len(image_paths) >= num_images:
                        break
            if len(image_paths) >= num_images:
                break
    
    if len(image_paths) == 0:
        print(f"❌ Error: No images found in {dataset_dir}")
        return
    
    print(f"📷 Found {len(image_paths)} images")
    
    # Create augmentation layer
    bg_aug = RandomBackgroundReplacement(prob=prob, face_center_ratio=face_ratio)
    
    # Process images
    original_images = []
    augmented_images = []
    
    for i, img_path in enumerate(image_paths[:num_images]):
        print(f"  Processing {i+1}/{min(len(image_paths), num_images)}: {os.path.basename(img_path)}")
        
        image = tf.keras.utils.load_img(img_path, target_size=(224, 224))
        image_array = tf.keras.utils.img_to_array(image) / 255.0
        image_batch = tf.expand_dims(image_array, 0)
        
        original_images.append(image_array)
        
        # Apply augmentation
        augmented = bg_aug(image_batch, training=True)
        augmented_images.append(augmented[0].numpy())
    
    # Create visualization
    fig, axes = plt.subplots(2, len(original_images), figsize=(5*len(original_images), 10))
    
    if len(original_images) == 1:
        axes = axes.reshape(2, 1)
    
    # Original images (top row)
    for i, img in enumerate(original_images):
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Original #{i+1}", fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
    
    # Augmented images (bottom row)
    for i, img in enumerate(augmented_images):
        axes[1, i].imshow(img)
        axes[1, i].set_title(f"Augmented #{i+1}\n(prob={prob})", fontsize=12)
        axes[1, i].axis('off')
    
    plt.suptitle("Adversarial Background Augmentation - Batch Test", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(output_dir, "adversarial_aug_batch.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved visualization to: {output_path}")
    
    plt.show()


def find_test_image(project_root):
    """Find a test image in the project directories"""
    test_paths = [
        os.path.join(project_root, "splitted_dataset", "train", "Drowsy"),
        os.path.join(project_root, "splitted_dataset", "train", "NotDrowsy"),
        os.path.join(project_root, "splitted_dataset", "val", "Drowsy"),
        os.path.join(project_root, "splitted_dataset", "val", "NotDrowsy"),
        os.path.join(project_root, "dataset", "Drowsy"),
        os.path.join(project_root, "dataset", "NotDrowsy"),
    ]
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for path in test_paths:
        if os.path.exists(path):
            for file in os.listdir(path):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    return os.path.join(path, file)
    return None


def find_test_dataset(project_root):
    """Find a test dataset directory"""
    test_paths = [
        os.path.join(project_root, "splitted_dataset", "train", "Drowsy"),
        os.path.join(project_root, "splitted_dataset", "train", "NotDrowsy"),
        os.path.join(project_root, "splitted_dataset", "val", "Drowsy"),
        os.path.join(project_root, "splitted_dataset", "val", "NotDrowsy"),
    ]
    
    for path in test_paths:
        if os.path.exists(path):
            return path
    return None


def main():
    """Main function with default paths"""
    print("=" * 60)
    print("🧪 Adversarial Background Augmentation Test Script")
    print("=" * 60)
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Default parameters
    output_dir = os.path.join(project_root, "test_outputs")
    prob = 1.0  # Always apply for testing
    face_ratio = 0.4
    num_samples = 6
    num_images = 4
    
    # Find test image
    test_image = find_test_image(project_root)
    test_dataset = find_test_dataset(project_root)
    
    if not test_image and not test_dataset:
        print("❌ Error: No test images found in project directories")
        print("   Please ensure dataset or splitted_dataset folders exist")
        return
    
    print(f"\n📁 Project root: {project_root}")
    print(f"📁 Output directory: {output_dir}")
    
    # Run all test modes
    print("\n" + "=" * 60)
    print("1️⃣  SINGLE IMAGE TEST")
    print("=" * 60)
    if test_image:
        print(f"   Using image: {test_image}")
        test_single_image(test_image, output_dir, prob, face_ratio)
    else:
        print("   ⚠️  No test image found, skipping...")
    
    print("\n" + "=" * 60)
    print("2️⃣  MULTIPLE AUGMENTATIONS TEST")
    print("=" * 60)
    if test_image:
        print(f"   Using image: {test_image}")
        print(f"   Generating {num_samples} different augmentations...")
        test_multiple_augmentations(test_image, output_dir, num_samples, face_ratio)
    else:
        print("   ⚠️  No test image found, skipping...")
    
    print("\n" + "=" * 60)
    print("3️⃣  BATCH TEST")
    print("=" * 60)
    if test_dataset:
        print(f"   Using dataset: {test_dataset}")
        print(f"   Testing {num_images} images...")
        test_batch_augmentation(test_dataset, output_dir, num_images, prob, face_ratio)
    else:
        print("   ⚠️  No test dataset found, skipping...")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    # Run with default paths - no arguments needed
    main()

