"""
Simple GradCAM Usage Example
"""
import numpy as np
import os
import tensorflow as tf
from src.model import build_model
from src.gradcam import GradCAM, analyze_model_gradcam
from src.dataloader import get_binary_pipelines


def main():
    """
    Simple GradCAM example for visualizing CNN attention.
    Supports both single image and batch-level analysis.
    """
    print("🎯 Simple GradCAM Example")
    print("=" * 40)
    
    # Adjustable parameters
    NUM_SAMPLES = 5          # Target number of examples to visualize
    MISCLASSIFIED_ONLY = False # If True, only show misclassified samples

    # Repository paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    #repo_root = os.path.dirname(project_root) #for file inside script folder

    # Dataset root directory (no train/val/test folders inside)
    dataset_root = os.path.join(project_root, "splitted_dataset")

    # 1. Load model
    print("1. Loading model...")
    RUN_NAME = "30_epoch-lib_sample-weight"
    MODEL_PATH = os.path.join(project_root, "runs", RUN_NAME, "models", "final_model.h5")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        print(f"✅ Loaded model: {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  Could not load model at {MODEL_PATH}: {e}")
        print("⚠️  Using a newly built (untrained) model for demonstration.")
        model = build_model()

    # 2. Load dataset
    print("2. Loading test dataset...")
    try:
        train_ds, val_ds, test_ds, class_names = get_binary_pipelines(dataset_root, use_masks=True)
        print("✅ Test dataset loaded successfully")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # 3. Single image GradCAM example
    print("3. Running single image GradCAM example...")
    for batch in train_ds.take(1):
        if isinstance(batch, (list, tuple)):
            if len(batch) == 2:
                batch_images, batch_labels = batch
            elif len(batch) == 3:
                batch_images, batch_labels, _ = batch
            else:
                raise ValueError(f"Unexpected batch format: {len(batch)} elements")
        else:
            raise ValueError("Unexpected batch type; expected (x,y) or (x,y,sample_weight)")
        sample_image = batch_images[0].numpy()
        true_label = int(batch_labels[0].numpy())
        break
    
    gradcam = GradCAM(model)
    fig, prediction = gradcam.visualize(
        sample_image, 
        class_names=['Not Drowsy', 'Drowsy'],
        save_path="gradcam_example.png"
    )
    
    pred_class = np.argmax(prediction)
    confidence = prediction[0][pred_class] if prediction.ndim == 2 else prediction[pred_class]
    print(f"   True: {['Not Drowsy', 'Drowsy'][true_label]}")
    print(f"   Pred: {['Not Drowsy', 'Drowsy'][pred_class]} ({confidence:.3f})")
    
    # 4. Batch-level GradCAM analysis
    print("4. Running batch GradCAM analysis...")
    # Derive output directory from model path
    output_dir = "gradcam_batch"
    model_dir = os.path.dirname(MODEL_PATH)
    if os.path.basename(model_dir) == "models":
        run_root = os.path.dirname(model_dir)
        output_dir = os.path.join(run_root, "plots", "gradcam_batch")
    else:
        # Fallback: if model path is inside runs/<run_name>/..., place results under that run
        parts = model_dir.replace("\\", "/").split("/")
        if "runs" in parts:
            runs_idx = parts.index("runs")
            if runs_idx + 1 < len(parts):
                run_root = os.path.join(*parts[:runs_idx+2])
                output_dir = os.path.join(run_root, "plots", "gradcam_batch")
    os.makedirs(output_dir, exist_ok=True)

    # Run GradCAM analysis
    try:
        analyze_model_gradcam(
            model,
            train_ds,
            num_samples=NUM_SAMPLES,
            output_dir=output_dir,
            class_names=class_names if class_names else ['Not Drowsy', 'Drowsy'],
            misclassified_only=MISCLASSIFIED_ONLY,
            #confusion=False  # You can set True to split into TP/TN/FP/FN
        )
    except Exception as e:
        print(f"⚠️  GradCAM analysis encountered an issue: {e}")
    
    # Extra info for misclassified-only mode
    if MISCLASSIFIED_ONLY:
        # Count how many samples were actually saved
        saved_images = [f for f in os.listdir(output_dir) if f.endswith('.png')]
        if len(saved_images) == 0:
            print("⚠️  No misclassified samples found. The dataset may be too accurate or too small.")
        elif len(saved_images) < NUM_SAMPLES:
            print(f"ℹ️  Only {len(saved_images)} misclassified samples found (requested {NUM_SAMPLES}).")

    print("\n✅ GradCAM example completed!")
    print(f"📁 Single example: gradcam_example.png")
    print(f"📁 Batch results: {output_dir}")


if __name__ == "__main__":
    main()
