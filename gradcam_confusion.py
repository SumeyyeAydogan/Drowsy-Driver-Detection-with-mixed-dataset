"""
Simple GradCAM implementation for drowsy driver detection
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D
try:
    # Optional conv variants
    from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
except Exception:  # pragma: no cover
    SeparableConv2D = tuple()  # type: ignore
    DepthwiseConv2D = tuple()  # type: ignore
import os
import re


# ---------- Helpers ----------

def _to_class_index(y) -> int:
    """
    Convert label tensors/arrays to a scalar class index.
    Works for:
      - binary scalar (float in {0.,1.})
      - shape (1,) binary
      - one-hot vector
    """
    arr = np.array(y)
    if arr.ndim == 0:
        return int(round(float(arr)))
    flat = arr.reshape(-1)
    if flat.size == 1:
        return int(round(float(flat[0])))
    return int(np.argmax(flat))


def _pred_to_prob_and_class(pred):
    """
    Normalize model outputs to:
      - prob: probability of class '1' (Drowsy)
      - cls: predicted class index (0/1)
    Supports:
      - sigmoid with shape (1,) or (1,1)
      - softmax with shape (1,2)
    """
    p = np.array(pred)
    # (1,1) or (1,)
    if p.ndim == 2 and p.shape[0] == 1 and p.shape[1] == 1:
        prob = float(p[0, 0]); return prob, int(prob >= 0.5)
    if p.ndim == 1 and p.shape[0] == 1:
        prob = float(p[0]);    return prob, int(prob >= 0.5)
    # (1,2) softmax
    if p.ndim == 2 and p.shape[0] == 1 and p.shape[1] == 2:
        prob = float(p[0, 1]); cls = int(np.argmax(p[0])); return prob, cls
    # fallback
    pr = float(p.ravel()[-1])
    return pr, int(pr >= 0.5)


# ---------- GradCAM ----------

class GradCAM:
    """
    Simple GradCAM for explaining CNN predictions
    """

    def __init__(self, model, layer_name=None):
        """
        Initialize GradCAM with a trained model
        """
        self.model = model

        # Build model once to ensure outputs exist
        if not hasattr(self.model, 'output') or self.model.output is None:
            dummy_input = tf.random.normal((1, 224, 224, 3))
            _ = self.model(dummy_input, training=False)

        # Pick last Conv2D layer if not provided
        self.layer_name = layer_name
        if self.layer_name is None:
            # Prefer last conv-like layer (Conv2D/Separable/Depthwise)
            conv_types = (Conv2D,)
            try:
                conv_types = (Conv2D, SeparableConv2D, DepthwiseConv2D)
            except Exception:
                pass
            for layer in reversed(self.model.layers):
                if isinstance(layer, conv_types):
                    self.layer_name = layer.name
                    break

        # Create grad model (conv outputs + final outputs)
        try:
            self.grad_model = Model(
                inputs=self.model.input,
                outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
            )
        except Exception:
            # Fallback: use original model; we'll return a dummy heatmap
            self.grad_model = self.model

    def compute_heatmap(self, image, class_idx=None):
        """
        Compute GradCAM heatmap.
        Works for:
        - Binary sigmoid (1 unit → probability of class 1)
        - Softmax (2 units)
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Fallback if grad_model couldn't be built
        if self.grad_model == self.model:
            prediction = self.model(image, training=False).numpy()
            h = 7 if image.shape[1] >= 7 else image.shape[1]
            w = 7 if image.shape[2] >= 7 else image.shape[2]
            heatmap = np.random.rand(h, w)
            heatmap /= (np.max(heatmap) + 1e-8)
            print(f"[GradCAM] Warning: Using fallback (dummy) heatmap for layer: {self.layer_name}")
            return heatmap, prediction

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image, training=False)
            predictions = tf.convert_to_tensor(predictions)

            # --- Determine which output neuron to explain ---
            if predictions.shape[-1] == 1:
                # Single sigmoid neuron → explain positive (class 1 / Drowsy)
                class_output = predictions[:, 0]
            else:
                # Two-class softmax → choose specified or predicted class
                if class_idx is None:
                    class_idx = tf.argmax(predictions[0]).numpy().item()
                class_output = predictions[:, class_idx]

        # --- Compute gradients ---
        grads = tape.gradient(class_output, conv_outputs)

        # Handle possible sequence models (5D tensor)
        if len(conv_outputs.shape) == 5:
            conv_outputs = conv_outputs[:, -1]
            grads = grads[:, -1]

        # Global average pooling over height & width
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]  # (H, W, C)

        # Weighted combination of channels
        heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=[[2], [0]])
        heatmap = tf.nn.relu(heatmap)
        heatmap /= (tf.reduce_max(heatmap) + 1e-8)

        return heatmap.numpy(), predictions.numpy()

    def overlay_heatmap(self, heatmap, image, alpha=0.4):
        """
        Overlay heatmap on image. Uses TF resize (no SciPy dependency).
        Input image can be uint8 [0..255] or float [0..1].
        """
        # Resize heatmap to match image size
        heatmap_tf = tf.convert_to_tensor(heatmap, dtype=tf.float32)
        heatmap_tf = heatmap_tf[None, ..., None]  # (1,H,W,1)
        H, W = int(image.shape[0]), int(image.shape[1])
        heatmap_resized = tf.image.resize(heatmap_tf, (H, W), method='bilinear')[0, ..., 0].numpy()

        # Normalize to [0,1]
        heatmap_norm = np.clip(heatmap_resized, 0.0, 1.0)
        img = image.astype(np.float32)
        if img.max() > 1.0:
            img = img / 255.0

        # Apply colormap (jet)
        cmap = plt.cm.get_cmap('jet')
        heatmap_colored = cmap(heatmap_norm)[..., :3]  # drop alpha

        # Blend
        overlayed = (1 - alpha) * img + alpha * heatmap_colored
        return np.clip(overlayed, 0.0, 1.0)

    def visualize(self, image, class_names=('Not Drowsy', 'Drowsy'),
                  threshold=0.5, target_class=None, save_path=None,
                  true_class_idx=None):
        """
        Create GradCAM visualization.
        - threshold: used to binarize sigmoid outputs
        - target_class: force CAM to a specific class (0/1 for softmax).
        """
        heatmap, prediction = self.compute_heatmap(
            image,
            class_idx=target_class
        )

        prob, pred_class = _pred_to_prob_and_class(prediction)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(image if image.max() <= 1.0 else image.astype(np.uint8))
        # Compose overlay text: Truth vs Prediction
        truth_text = None
        if true_class_idx is not None and 0 <= int(true_class_idx) < len(class_names):
            truth_text = f"Truth: {class_names[int(true_class_idx)]}"
        pred_text = f"Pred: {class_names[pred_class]} ({prob:.3f})"
        overlay_text = pred_text if truth_text is None else f"{truth_text} | {pred_text}"
        # Draw readable label on the image
        axes[0].text(
            5, 15, overlay_text,
            color='white', fontsize=10,
            bbox=dict(facecolor='black', alpha=0.6, edgecolor='none')
        )
        axes[0].set_title('Original')
        axes[0].axis('off')

        # Heatmap
        im1 = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title('GradCAM Heatmap')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1])

        # Overlay
        overlayed = self.overlay_heatmap(heatmap, image)
        axes[2].imshow(overlayed)
        axes[2].set_title('GradCAM Overlay')
        axes[2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GradCAM saved to: {save_path}")

        return fig, prediction


def analyze_model_gradcam(model, test_ds, num_samples=10, output_dir="gradcam_results",
                         class_names=('Not Drowsy', 'Drowsy'), threshold=0.5,
                         subject_diverse_dir=None, misclassified_only=False,
                         confusion=False, max_per_category=10,
                         confusion_limit=False):
    """
    Analyze model with GradCAM on samples from test_ds.
    Supports:
      - confusion=True: saves TP/TN/FP/FN folders.
      - misclassified_only=True: saves only FP/FN up to max_per_category each.
      - confusion_limit=True: applies max_per_category limit to each confusion bucket.
    Gracefully handles cases with fewer available samples.
    """

    # Prepare output folders
    os.makedirs(output_dir, exist_ok=True)
    if confusion:
        for sub in ("TP", "TN", "FP", "FN"):
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    elif misclassified_only:
        for sub in ("FP", "FN"):
            os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    gradcam = GradCAM(model)
    sample_count = 0

    # Counters for each category
    fp_saved, fn_saved, tp_saved, tn_saved = 0, 0, 0, 0
    batch_count = 0
    max_batches_to_check = 50  # Maksimum 50 batch kontrol et

    for batch_data in test_ds:
        batch_count += 1
        
        # Maksimum batch limitini kontrol et
        if batch_count > max_batches_to_check:
            print(f"⚠️  Reached maximum batch limit ({max_batches_to_check}). Stopping.")
            print(f"📊 Final counts: TP={tp_saved}, TN={tn_saved}, FP={fp_saved}, FN={fn_saved}")
            break
        # Handle both formats: (x, y) or (x, y, sample_weight)
        if len(batch_data) == 2:
            batch_images, batch_labels = batch_data
        elif len(batch_data) == 3:
            batch_images, batch_labels, sample_weight = batch_data
        else:
            raise ValueError(f"Unexpected batch format: {len(batch_data)} elements")
        
        # Stop early if all requested categories reached their limits
        if misclassified_only and fp_saved >= max_per_category and fn_saved >= max_per_category:
            print(f"✅ Found enough misclassified samples: FP={fp_saved}, FN={fn_saved}")
            break
        
        # For misclassified-only: if we've checked many batches and found some samples, stop
        if misclassified_only and batch_count > 20 and (fp_saved + fn_saved) >= 3:
            print(f"⚠️  Found {fp_saved + fn_saved} misclassified samples after {batch_count} batches. Stopping.")
            break
        if confusion and confusion_limit and all([
            tp_saved >= max_per_category,
            tn_saved >= max_per_category,
            fp_saved >= max_per_category,
            fn_saved >= max_per_category
        ]):
            print(f"✅ Found enough samples for all categories: TP={tp_saved}, TN={tn_saved}, FP={fp_saved}, FN={fn_saved}")
            break
        if not misclassified_only and not confusion and sample_count >= num_samples:
            print(f"✅ Found enough general samples: {sample_count}")
            break

        batch_preds = model.predict(batch_images, verbose=0)

        # Check if we can skip this entire batch
        if confusion and confusion_limit:
            # Count categories in this batch first
            batch_tp = batch_tn = batch_fp = batch_fn = 0
            for i in range(len(batch_images)):
                true_idx = _to_class_index(batch_labels[i].numpy())
                local_pred = batch_preds[i:i+1]
                _, pred_idx_local = _pred_to_prob_and_class(local_pred)
                
                if int(true_idx) == 1 and pred_idx_local == 1:
                    batch_tp += 1
                elif int(true_idx) == 0 and pred_idx_local == 0:
                    batch_tn += 1
                elif int(true_idx) == 0 and pred_idx_local == 1:
                    batch_fp += 1
                else:
                    batch_fn += 1
            
            # Skip batch if we already have enough of each category
            if (tp_saved >= max_per_category and batch_tp == 0) and \
               (tn_saved >= max_per_category and batch_tn == 0) and \
               (fp_saved >= max_per_category and batch_fp == 0) and \
               (fn_saved >= max_per_category and batch_fn == 0):
                print(f"⏭️  Skipping batch - no needed categories")
                continue

        for i in range(len(batch_images)):
            image = batch_images[i].numpy()
            true_idx = _to_class_index(batch_labels[i].numpy())

            local_pred = batch_preds[i:i+1]
            prob_local, pred_idx_local = _pred_to_prob_and_class(local_pred)

            # Determine confusion matrix category
            if int(true_idx) == 1 and pred_idx_local == 1:
                bucket = "TP"
            elif int(true_idx) == 0 and pred_idx_local == 0:
                bucket = "TN"
            elif int(true_idx) == 0 and pred_idx_local == 1:
                bucket = "FP"
            else:
                bucket = "FN"

            # Skip non-misclassified if requested
            if misclassified_only and bucket not in ("FP", "FN"):
                continue

            # Apply per-category limits (for misclassified or confusion_limit modes)
            if misclassified_only or (confusion and confusion_limit):
                if bucket == "FP" and fp_saved >= max_per_category:
                    continue
                if bucket == "FN" and fn_saved >= max_per_category:
                    continue
                if confusion_limit:
                    if bucket == "TP" and tp_saved >= max_per_category:
                        continue
                    if bucket == "TN" and tn_saved >= max_per_category:
                        continue

            # Stop entirely if general limit reached
            if not misclassified_only and not confusion and sample_count >= num_samples:
                break

            # Compute GradCAM
            target_class = None
            if np.array(batch_preds).ndim == 2 and np.array(batch_preds).shape[1] == 2:
                target_class = 1  # explain "Drowsy" class

            fig, pred_vec = gradcam.visualize(
                image,
                class_names=class_names,
                threshold=threshold,
                target_class=target_class,
                true_class_idx=true_idx,
                save_path=None
            )

            prob, pred_idx = _pred_to_prob_and_class(pred_vec)

            # Save visualization
            save_folder = os.path.join(output_dir, bucket if (confusion or misclassified_only) else "")
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f'sample_{sample_count:03d}.png')
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Sample {sample_count:03d}: True={class_names[true_idx]}, "
                  f"Pred={class_names[pred_idx]} ({prob:.3f}) -> {bucket}")

            # Update counters
            if bucket == "FP":
                fp_saved += 1
            elif bucket == "FN":
                fn_saved += 1
            elif bucket == "TP":
                tp_saved += 1
            elif bucket == "TN":
                tn_saved += 1

            sample_count += 1

        # Break outer loop if limits reached
        if misclassified_only and fp_saved >= max_per_category and fn_saved >= max_per_category:
            break
        if confusion and confusion_limit and all([
            tp_saved >= max_per_category,
            tn_saved >= max_per_category,
            fp_saved >= max_per_category,
            fn_saved >= max_per_category
        ]):
            break

    # Summary reporting
    if misclassified_only:
        print("\n--- Misclassified Summary ---")
        print(f"FP saved: {fp_saved}/{max_per_category}")
        print(f"FN saved: {fn_saved}/{max_per_category}")
        if fp_saved == 0 and fn_saved == 0:
            print("⚠️ No FP or FN samples found.")
        else:
            if fp_saved < max_per_category:
                print(f"ℹ️ Only {fp_saved} FP samples found (requested {max_per_category}).")
            if fn_saved < max_per_category:
                print(f"ℹ️ Only {fn_saved} FN samples found (requested {max_per_category}).")
    elif confusion and confusion_limit:
        print("\n--- Confusion Matrix Summary ---")
        print(f"TP saved: {tp_saved}/{max_per_category}")
        print(f"TN saved: {tn_saved}/{max_per_category}")
        print(f"FP saved: {fp_saved}/{max_per_category}")
        print(f"FN saved: {fn_saved}/{max_per_category}")
        print(f"✅ GradCAM (confusion_limit mode) complete! Results saved in: {output_dir}/")
    else:
        print(f"\n✅ GradCAM analysis complete! Results saved in: {output_dir}/")
