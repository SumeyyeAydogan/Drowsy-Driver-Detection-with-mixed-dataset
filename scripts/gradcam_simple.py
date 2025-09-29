"""
Simple GradCAM Usage Example
"""
import numpy as np
import tensorflow as tf
from src.model import build_model
from src.gradcam import GradCAM, analyze_model_gradcam
from src.dataloader import get_data_pipelines


def main():
    """
    Simple GradCAM example
    """
    print("🎯 Simple GradCAM Example")
    print("=" * 40)
    
    # 1. Load or create model
    print("1. Loading model...")
    try:
        # Try to load a trained model
        model = tf.keras.models.load_model("runs/100_epoch/100_epoch.zip")
        print("✅ Loaded trained model")
    except:
        # Create new model if no trained model found
        model = build_model()
        print("⚠️  Using untrained model")
    
    # 2. Load test data
    print("2. Loading test data...")
    try:
        train_ds, val_ds, test_ds = get_binary_pipelines("dataset")
        print("✅ Test data loaded")
    except:
        print("❌ Error loading test data")
        return
    
    # 3. Single image example
    print("3. Single image GradCAM...")
    for batch_images, batch_labels in test_ds.take(1):
        sample_image = batch_images[0].numpy()
        true_label = batch_labels[0].numpy()
        break
    
    gradcam = GradCAM(model)
    fig, prediction = gradcam.visualize(
        sample_image, 
        class_names=['Not Drowsy', 'Drowsy'],
        save_path="gradcam_example.png"
    )
    
    pred_class = np.argmax(prediction)
    confidence = prediction[pred_class]
    print(f"   True: {['Not Drowsy', 'Drowsy'][true_label]}")
    print(f"   Pred: {['Not Drowsy', 'Drowsy'][pred_class]} ({confidence:.3f})")
    
    # 4. Batch analysis
    print("4. Batch analysis...")
    analyze_model_gradcam(model, test_ds, num_samples=3, output_dir="gradcam_batch")
    
    print("\n✅ GradCAM example completed!")
    print("📁 Check 'gradcam_example.png' and 'gradcam_batch/' directory")


if __name__ == "__main__":
    main()
