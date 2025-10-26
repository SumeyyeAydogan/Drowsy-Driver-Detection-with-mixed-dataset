# callbacks.py - Custom training callbacks with GradCAM visualizations
import tensorflow as tf
import os
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.gradcam import GradCAM  # Ensure your GradCAM class is imported

# ------------------------------
# 1. Checkpoint callback
# ------------------------------
class CheckpointCallback(tf.keras.callbacks.Callback):
    """Custom callback to save checkpoints and metrics after each epoch"""
    
    def __init__(self, run_manager):
        super().__init__()
        self.run_manager = run_manager
        
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch"""
        epoch_num = epoch + 1  # Keras epochs are 0-indexed
        
        # Save checkpoint
        self.run_manager.save_checkpoint(self.model, epoch_num)
        
        # Save metrics
        self.run_manager.save_metrics(self.model.history, epoch_num)
        
        print(f"✅ Epoch {epoch_num} completed and checkpoint/metrics saved!")

# ------------------------------
# 2. GradCAM visualization callback
# ------------------------------
class GradCAMEpochCallback(tf.keras.callbacks.Callback):
    """
    Callback to save GradCAM visualizations for validation dataset at the end of each epoch.
    Visualizations are saved into TP/TN/FP/FN folders.
    """
    def __init__(self, test_ds, output_dir="gradcam_epoch_outputs", max_samples=5):
        """
        Args:
            test_ds: tf.data.Dataset for GradCAM visualization (validation set recommended)
            output_dir: directory to save GradCAM images
            max_samples: max number of samples per epoch to save
        """
        super().__init__()
        self.test_ds = test_ds
        self.output_dir = output_dir
        self.max_samples = max_samples
        os.makedirs(output_dir, exist_ok=True)
        self.gradcam = None

    def on_epoch_end(self, epoch, logs=None):
        epoch_num = epoch + 1
        print(f"\n[GradCAM] Saving visualizations for epoch {epoch_num}...")
        epoch_dir = os.path.join(self.output_dir, f"epoch_{epoch_num:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Create TP/TN/FP/FN folders
        tp_dir = os.path.join(epoch_dir, "TP")
        tn_dir = os.path.join(epoch_dir, "TN")
        fp_dir = os.path.join(epoch_dir, "FP")
        fn_dir = os.path.join(epoch_dir, "FN")
        for d in [tp_dir, tn_dir, fp_dir, fn_dir]:
            os.makedirs(d, exist_ok=True)

        if self.gradcam is None:
            self.gradcam = GradCAM(self.model)

        sample_count = 0
        for batch_images, batch_labels in self.test_ds:
            for i in range(len(batch_images)):
                if sample_count >= self.max_samples:
                    break
                image = batch_images[i].numpy()
                # Determine true class index (binary mode: single float value)
                # Label is already 0.0 or 1.0 in binary mode
                true_idx = int(batch_labels[i].numpy())

                # Model prediction
                pred_vec = self.model.predict(image[None, ...], verbose=0)
                # Model outputs sigmoid probability [0.0-1.0]
                pred_prob = float(pred_vec.ravel()[0])
                pred_idx = 1 if pred_prob >= 0.5 else 0

                # Select folder based on TP/TN/FP/FN
                if true_idx == 1 and pred_idx == 1:
                    folder = tp_dir
                    status = "TP"
                elif true_idx == 0 and pred_idx == 0:
                    folder = tn_dir
                    status = "TN"
                elif true_idx == 0 and pred_idx == 1:
                    folder = fp_dir
                    status = "FP"
                elif true_idx == 1 and pred_idx == 0:
                    folder = fn_dir
                    status = "FN"

                # File path
                save_path = os.path.join(folder, f"sample_{sample_count:02d}_true{true_idx}_pred{pred_idx}.png")
                
                # Debug print
                if sample_count < 3:  # Print first 3 samples
                    print(f"  Sample {sample_count}: True={true_idx}, Pred={pred_idx} (prob={pred_prob:.3f}) -> {status}")

                # Save GradCAM visualization
                self.gradcam.visualize(image, save_path=save_path, true_class_idx=true_idx)

                sample_count += 1

            if sample_count >= self.max_samples:
                break

        print(f"[GradCAM] Saved {sample_count} GradCAM samples for epoch {epoch_num}")

# ------------------------------
# 3. Function to get all training callbacks
# ------------------------------
def get_training_callbacks(run_manager, val_ds=None, gradcam_output_dir="gradcam_epoch_outputs", max_samples=5):
    """
    Returns all training callbacks including checkpoint, early stopping,
    learning rate scheduler, and optional GradCAM visualizations.

    Args:
        run_manager: RunManager instance for checkpoint/metrics
        val_ds: tf.data.Dataset for GradCAM visualization (validation set recommended)
        gradcam_output_dir: folder to save GradCAM outputs
        max_samples: max number of samples per epoch to save

    Returns:
        list of callbacks
    """
    callbacks = [
        CheckpointCallback(run_manager),
        #EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6)
    ]

    
    if val_ds is not None:
        callbacks.append(
            GradCAMEpochCallback(test_ds=val_ds, output_dir=gradcam_output_dir, max_samples=max_samples)
        )

    return callbacks
