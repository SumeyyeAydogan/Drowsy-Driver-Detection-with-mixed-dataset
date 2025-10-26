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
        - For binary sigmoid (1 unit): class_idx is forced to 0.
        - For 2-class softmax: class_idx can be 0/1; defaults to argmax.
        """
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        # Fallback path if grad_model could not be built
        if self.grad_model == self.model:
            prediction = self.model(image, training=False).numpy()
            # Produce a small dummy heatmap just to keep the pipeline running
            h = 7 if image.shape[1] >= 7 else image.shape[1]
            w = 7 if image.shape[2] >= 7 else image.shape[2]
            heatmap = np.random.rand(h, w)
            heatmap = heatmap / (np.max(heatmap) + 1e-8)
            print("[GradCAM] Warning: Using fallback (dummy) heatmap. Could not build grad model for layer:", self.layer_name)
            return heatmap, prediction

        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(image, training=False)
            # Decide which logit/probability to explain
            predictions = tf.convert_to_tensor(predictions)
            if predictions.shape[-1] == 1:
                # Binary sigmoid: single neuron (prob of class 1)
                class_idx = 0  # the only column
                class_output = predictions[:, 0]
            else:
                # 2-class softmax
                if class_idx is None:
                    class_idx = tf.argmax(predictions[0]).numpy().item()
                class_output = predictions[:, class_idx]

        grads = tape.gradient(class_output, conv_outputs)
        # Global average pooling over H,W
        # Support possible time dimension: (B,T,H,W,C)
        if len(conv_outputs.shape) == 5:
            # take last time-step features
            conv_outputs = conv_outputs[:, -1]
            grads = grads[:, -1]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]  # (H, W, C)

        # Weighted sum across channels
        heatmap = tf.tensordot(conv_outputs, pooled_grads, axes=[[2], [0]])
        heatmap = tf.nn.relu(heatmap)
        denom = tf.reduce_max(heatmap)
        heatmap = heatmap / (denom + 1e-8)

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

    def visualize(self, image, class_names=('NotDrowsy', 'Drowsy'),  # ✅ Match dataloader (no space)
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


def analyze_subjects_gradcam(
    model,
    test_dir,
    num_samples=10,
    output_dir="gradcam_subjects",
    class_names=("NotDrowsy", "Drowsy"),
    img_size=(224, 224),
    seed=42
):
    """
    GradCAM analysis by subjects (case-sensitive).

    - Loads images from a test directory containing 'Drowsy' and 'NotDrowsy' subfolders.
    - Picks one image per subject (case-sensitive; e.g., 'ALICE' ≠ 'alice').
    - Generates and saves GradCAM visualizations grouped into TP, TN, FP, FN folders.
    """

    rng = np.random.default_rng(seed)

    # 1️⃣ Build dataset automatically from directory (labels inferred)
    ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="binary",  # ✅ Match dataloader label_mode
        class_names=list(class_names),
        image_size=img_size,
        shuffle=False
    )

    file_paths = ds.file_paths  # all file paths in the dataset

    ds = ds.apply(tf.data.experimental.ignore_errors())

    # 2️⃣ Extract subject names (case-sensitive)
    subj_re = re.compile(r"^([A-Za-z]+)")
    subj_to_examples = {}

    for path, (img, label) in zip(file_paths, ds.unbatch()):
        fname = os.path.basename(path)
        m = subj_re.match(fname)
        if not m:
            continue
        subj = m.group(1)  # case-sensitive (do NOT lowercase)
        subj_to_examples.setdefault(subj, []).append((path, int(label.numpy())))

    subjects = list(subj_to_examples.keys())
    rng.shuffle(subjects)

    print(f"📂 Found {len(subjects)} subjects (case-sensitive).")

    # 3️⃣ Create output folders for each confusion category
    for sub in ("TP", "TN", "FP", "FN"):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

    gradcam = GradCAM(model)
    sample_count = 0

    # 4️⃣ Iterate over subjects and analyze
    for subj in subjects:
        if sample_count >= num_samples:
            break

        examples = subj_to_examples[subj]
        fpath, true_idx = examples[rng.integers(0, len(examples))]

        # Load and normalize the image
        img = tf.keras.utils.load_img(fpath, target_size=img_size)
        img_arr = tf.keras.utils.img_to_array(img) / 255.0

        # Model prediction
        pred_vec = model.predict(img_arr[None, ...], verbose=0)
        prob, pred_idx = _pred_to_prob_and_class(pred_vec)

        # Determine confusion category
        if true_idx == 1 and pred_idx == 1:
            bucket = "TP"
        elif true_idx == 0 and pred_idx == 0:
            bucket = "TN"
        elif true_idx == 0 and pred_idx == 1:
            bucket = "FP"
        else:
            bucket = "FN"

        # Generate and save GradCAM visualization
        fig, _ = gradcam.visualize(
            img_arr,
            class_names=class_names,
            true_class_idx=true_idx,
            save_path=os.path.join(output_dir, bucket, f"{subj}.png")
        )
        plt.close(fig)

        print(f"🧍 {subj}: True={class_names[true_idx]}, Pred={class_names[pred_idx]} "
              f"({prob:.2f}) -> {bucket}")
        sample_count += 1

    print(f"\n✅ GradCAM analysis completed ({sample_count} subjects processed).")
    print(f"Results saved in: {output_dir}/[TP|TN|FP|FN]/")