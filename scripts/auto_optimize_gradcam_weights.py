"""
Auto-optimize GradCAM sample weight parameters
with consistent eye+mouth ROI mask (same as training).

IMPROVEMENTS:
1. ✅ Uses same mask as training (src/simple_mask.py)
2. ✅ Better heatmap resize (bilinear interpolation)
3. ✅ Dynamic parameters based on data distribution
4. ✅ Weight analysis with histogram and statistics

Outputs:
- optimized_gradcam_weights.json  (weights per image)
- gradcam_opt_params.json         (best params)
- gradcam_opt_report.txt          (summary report)
- gradcam_weight_histogram.png    (weight distribution)
"""

import os, json, sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from statistics import median
import matplotlib.pyplot as plt

# Add root path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gradcam import CustomGradCAM
from src.simple_mask import create_mask_numpy


# ================== CONFIG ======================
CONFIG = {
    "model_path": r"runs/30_epoch_without-mask_sbj-gradcam-fixed/models/final_model.h5",
    "data_dir": r"splitted_dataset/train",
    "img_size": (224, 224),
    "use_soft_mask": False,  # Match training mask type
    "mask_alpha": 0.2,        # Match training mask alpha
    "apply_gaussian": True,   # Apply Gaussian blur for smoother mask
    "gaussian_sigma": 7.0,    # Gaussian blur sigma
    "search_level": 2,        # (2 = ORTA)
    "weight_mode": "penalize"   # "reward" or "penalize"
    # "reward": High focus ratio → High weight (reward good behavior)
    # "penalize": Low focus ratio → High weight (penalize bad behavior)
}


# ================== MASK ======================
def create_face_mask(img_size, use_soft_mask=None, mask_alpha=None, apply_gaussian=None, gaussian_sigma=None):
    """
    Create mask using same logic as training (src/simple_mask.py).
    This ensures consistency between optimization and training.
    """
    # Use config defaults if not provided
    use_soft_mask = use_soft_mask if use_soft_mask is not None else CONFIG["use_soft_mask"]
    mask_alpha = mask_alpha if mask_alpha is not None else CONFIG["mask_alpha"]
    apply_gaussian = apply_gaussian if apply_gaussian is not None else CONFIG["apply_gaussian"]
    gaussian_sigma = gaussian_sigma if gaussian_sigma is not None else CONFIG["gaussian_sigma"]
    
    return create_mask_numpy(
        img_size=img_size,
        use_soft_mask=use_soft_mask,
        alpha=mask_alpha,
        apply_gaussian=apply_gaussian,
        sigma=gaussian_sigma
    )

"""
def create_face_mask(img_size, face_center_ratio=0.75):
    h, w = img_size
    center_h = int(h * face_center_ratio)
    center_w = int(w * face_center_ratio)
    y0 = (h - center_h) // 2
    x0 = (w - center_w) // 2
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y0+center_h, x0:x0+center_w] = 1.0
    return mask
"""

def compute_focus_ratio(heatmap, mask):
    """
    Compute focus ratio: how much of the heatmap is in the ROI mask.
    
    Args:
        heatmap: Normalized heatmap (0-1)
        mask: ROI mask (0-1)
    
    Returns:
        Focus ratio (0-1): higher = more focus on ROI
    """
    heatmap = np.maximum(heatmap, 0)  # Ensure non-negative
    if heatmap.max() > 0:
        heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize to 0-1
    focus = np.sum(heatmap * mask)      # ROI'deki toplam heatmap
    total = np.sum(heatmap) + 1e-8      # Tüm heatmap toplamı
    return float(focus / total)


# ================== CORE ======================
def collect_focus_distribution(model, data_dir, img_size):
    """
    Collect focus ratios for all images using consistent mask.
    
    IMPROVEMENT: Better heatmap resize with bilinear interpolation.
    """
    gradcam = CustomGradCAM(model)

    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, labels="inferred", label_mode="binary",
        image_size=img_size, batch_size=1, shuffle=False
    )
    file_paths = list(getattr(ds, "file_paths", []))
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths).batch(1)
    ds = tf.data.Dataset.zip((ds, path_ds))
    ds = ds.apply(tf.data.experimental.ignore_errors())

    # Use consistent mask (same as training)
    mask = create_face_mask(img_size)
    ratios = []

    print("[AutoOpt] Computing focus distribution...")
    print(f"[AutoOpt] Mask type: {'soft' if CONFIG['use_soft_mask'] else 'hard'}, "
          f"Gaussian: {CONFIG['apply_gaussian']}")

    for idx, (data_batch, path_batch) in enumerate(ds):
        images, labels = data_batch
        image = images[0].numpy() / 255.0
        label = int(labels[0].numpy())

        heatmap = gradcam.compute_heatmap(image, class_idx=label)
        
        # IMPROVEMENT: Better resize with bilinear interpolation
        # Resize heatmap to match image size with better quality
        heatmap = tf.image.resize(
            heatmap[..., None], 
            img_size, 
            method='bilinear',  # Better than default
            antialias=True      # Anti-aliasing for smoother resize
        ).numpy()[..., 0]

        ratio = compute_focus_ratio(heatmap, mask)
        ratios.append(ratio)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(file_paths)}")

    return ratios, file_paths


# ================== OPTIMIZATION ======================
def choose_params(ratios, level):
    """
    IMPROVEMENT: Dynamic parameter optimization based on data distribution.
    
    Optimizes:
    - target_focus (based on focus distribution with std consideration)
    - alpha (based on virtual penalty sim with more candidates)
    - clip range (scaled from alpha with better bounds)
    """
    ratios = np.array(ratios)
    ratios_sorted = sorted(ratios)
    med = median(ratios)
    mean_r = float(np.mean(ratios))
    std_r = float(np.std(ratios))
    q25 = float(np.percentile(ratios, 25))
    q75 = float(np.percentile(ratios, 75))

    print(f"[AutoOpt] Focus Ratio Stats: mean={mean_r:.3f}, median={med:.3f}, "
          f"std={std_r:.3f}, Q25={q25:.3f}, Q75={q75:.3f}")

    # --------- IMPROVEMENT: Dynamic Target Focus ---------
    # Use median + adaptive offset based on std
    # If std is high, use smaller offset (more conservative)
    # If std is low, use larger offset (more aggressive)
    adaptive_offset = min(0.15, max(0.05, std_r * 0.5))
    target_focus = min(0.92, max(0.45, med + adaptive_offset))
    
    print(f"[AutoOpt] Adaptive offset: {adaptive_offset:.3f}, Target focus: {target_focus:.3f}")

    # --------- IMPROVEMENT: Extended Alpha Testing ---------
    # More candidates for better optimization
    candidates = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]

    def simulate_penalty(alpha):
        penalties = [abs(target_focus - r) * alpha for r in ratios]
        return np.mean(penalties)

    penalties = {a: simulate_penalty(a) for a in candidates}
    best_alpha = min(penalties, key=penalties.get)
    
    print(f"[AutoOpt] Alpha candidates penalties: {penalties}")
    print(f"[AutoOpt] Best alpha: {best_alpha}")

    # Avoid extreme alpha for stability
    if best_alpha > 1.2:
        best_alpha = 1.2
    if best_alpha < 0.4:
        best_alpha = 0.4

    # --------- IMPROVEMENT: Better Clip Range ---------
    # More conservative clipping based on distribution
    clip_range_factor = best_alpha * 0.6  # Slightly more conservative
    clip_min = 1.0 - clip_range_factor
    clip_max = 1.0 + clip_range_factor

    # Ensure reasonable bounds
    clip_min = max(0.5, clip_min)
    clip_max = min(2.0, clip_max)  # Allow more range if needed

    return dict(
        target_focus=float(target_focus),
        alpha=float(best_alpha),
        clip_min=float(clip_min),
        clip_max=float(clip_max),
        # Additional stats for reporting
        mean_focus=float(mean_r),
        median_focus=float(med),
        std_focus=float(std_r),
        q25_focus=float(q25),
        q75_focus=float(q75),
    )


# ================== APPLY ======================
def apply_weights(params, ratios, file_paths, output_path):
    """
    Apply weights to each image based on focus ratio.
    
    Two modes:
    1. "reward": High focus ratio → High weight (reward good ROI focus)
    2. "penalize": Low focus ratio → High weight (penalize bad ROI focus)
    
    IMPROVEMENT: Returns weight statistics for analysis.
    """
    weights = {}
    weight_values = []
    mode = CONFIG.get("weight_mode", "reward")

    print(f"\n[AutoOpt] Applying weights (mode: {mode})...")

    for r, fp in zip(ratios, file_paths):
        if mode == "reward":
            # Reward approach: High focus ratio → High weight
            # delta = r - target_focus
            # If r > target_focus → delta > 0 → w > 1 (reward)
            # If r < target_focus → delta < 0 → w < 1 (penalize)
            delta = r - params["target_focus"]
        else:  # "penalize"
            # Penalize approach: Low focus ratio → High weight
            # delta = target_focus - r
            # If r < target_focus → delta > 0 → w > 1 (penalize)
            # If r > target_focus → delta < 0 → w < 1 (reward)
            delta = params["target_focus"] - r
        
        w = 1 + params["alpha"] * delta
        w = float(np.clip(w, params["clip_min"], params["clip_max"]))
        
        weight_values.append(w)
        rel = os.path.relpath(fp, CONFIG["data_dir"]).replace("\\", "/")
        weights[rel] = w

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(weights, f, indent=2)

    # Calculate statistics
    weight_stats = {
        "mean": float(np.mean(weight_values)),
        "median": float(np.median(weight_values)),
        "std": float(np.std(weight_values)),
        "min": float(np.min(weight_values)),
        "max": float(np.max(weight_values)),
        "q25": float(np.percentile(weight_values, 25)),
        "q75": float(np.percentile(weight_values, 75)),
    }
    
    print(f"[AutoOpt] Weight Stats: mean={weight_stats['mean']:.3f}, "
          f"median={weight_stats['median']:.3f}, std={weight_stats['std']:.3f}")
    print(f"[AutoOpt] Weight Range: [{weight_stats['min']:.3f}, {weight_stats['max']:.3f}]")

    return weights, weight_values, weight_stats


# ================== ANALYSIS ======================
def plot_weight_histogram(weight_values, ratios, params, output_path):
    """
    IMPROVEMENT: Create histogram and analysis plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Weight distribution histogram
    axes[0, 0].hist(weight_values, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(weight_values), color='r', linestyle='--', label=f'Mean: {np.mean(weight_values):.3f}')
    axes[0, 0].axvline(np.median(weight_values), color='g', linestyle='--', label=f'Median: {np.median(weight_values):.3f}')
    axes[0, 0].set_xlabel('Weight Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Weight Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Focus ratio distribution
    axes[0, 1].hist(ratios, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].axvline(params["target_focus"], color='r', linestyle='--', 
                       label=f'Target: {params["target_focus"]:.3f}')
    axes[0, 1].axvline(np.median(ratios), color='g', linestyle='--', 
                       label=f'Median: {np.median(ratios):.3f}')
    axes[0, 1].set_xlabel('Focus Ratio')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Focus Ratio Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Weight vs Focus Ratio scatter
    axes[1, 0].scatter(ratios, weight_values, alpha=0.3, s=10)
    axes[1, 0].axvline(params["target_focus"], color='r', linestyle='--', 
                       label=f'Target: {params["target_focus"]:.3f}')
    axes[1, 0].set_xlabel('Focus Ratio')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].set_title('Weight vs Focus Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Weight statistics text
    stats_text = f"""
Weight Statistics:
  Mean: {np.mean(weight_values):.3f}
  Median: {np.median(weight_values):.3f}
  Std: {np.std(weight_values):.3f}
  Min: {np.min(weight_values):.3f}
  Max: {np.max(weight_values):.3f}
  Q25: {np.percentile(weight_values, 25):.3f}
  Q75: {np.percentile(weight_values, 75):.3f}

Focus Ratio Statistics:
  Mean: {np.mean(ratios):.3f}
  Median: {np.median(ratios):.3f}
  Std: {np.std(ratios):.3f}
  Target: {params["target_focus"]:.3f}

Parameters:
  Alpha: {params["alpha"]:.3f}
  Clip Range: [{params["clip_min"]:.3f}, {params["clip_max"]:.3f}]
"""
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                    verticalalignment='center', transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[AutoOpt] Histogram saved: {output_path}")


# ================== MAIN ======================
if __name__ == "__main__":
    cfg = CONFIG

    model = tf.keras.models.load_model(cfg["model_path"], compile=False)

    ratios, file_paths = collect_focus_distribution(
        model, cfg["data_dir"], cfg["img_size"])

    params = choose_params(ratios, cfg["search_level"])
    print("\n[AutoOpt] Best Params:", {k: v for k, v in params.items() 
                                       if k not in ['mean_focus', 'median_focus', 'std_focus', 'q25_focus', 'q75_focus']})

    # SAVE PARAMS
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/gradcam_opt_params.json", "w") as f:
        json.dump(params, f, indent=2)

    # APPLY & SAVE
    weights, weight_values, weight_stats = apply_weights(
        params, ratios, file_paths,
        output_path="artifacts/optimized_gradcam_weights.json")

    # IMPROVEMENT: Create histogram
    plot_weight_histogram(
        weight_values, ratios, params,
        output_path="artifacts/gradcam_weight_histogram.png"
    )

    # IMPROVEMENT: Enhanced report
    with open("artifacts/gradcam_opt_report.txt", "w") as f:
        f.write("=== GradCAM Optimization Report (IMPROVED) ===\n\n")
        f.write("Weight Mode:\n")
        mode = cfg.get("weight_mode", "reward")
        f.write(f"  Mode: {mode}\n")
        if mode == "reward":
            f.write("  Strategy: High focus ratio → High weight (reward good ROI focus)\n")
        else:
            f.write("  Strategy: Low focus ratio → High weight (penalize bad ROI focus)\n")
        f.write("\nFocus Ratio Statistics:\n")
        f.write(f"  Mean: {params.get('mean_focus', np.mean(ratios)):.3f}\n")
        f.write(f"  Median: {params.get('median_focus', median(ratios)):.3f}\n")
        f.write(f"  Std: {params.get('std_focus', np.std(ratios)):.3f}\n")
        f.write(f"  Q25: {params.get('q25_focus', np.percentile(ratios, 25)):.3f}\n")
        f.write(f"  Q75: {params.get('q75_focus', np.percentile(ratios, 75)):.3f}\n\n")
        f.write("Optimized Parameters:\n")
        f.write(f"  Target Focus: {params['target_focus']:.3f}\n")
        f.write(f"  Alpha: {params['alpha']:.3f}\n")
        f.write(f"  Clip Range: [{params['clip_min']:.3f}, {params['clip_max']:.3f}]\n\n")
        f.write("Weight Statistics:\n")
        for key, value in weight_stats.items():
            f.write(f"  {key.capitalize()}: {value:.3f}\n")
        f.write(f"\nTotal Samples: {len(ratios)}\n")
        f.write(f"\nMask Configuration:\n")
        f.write(f"  Use Soft Mask: {cfg['use_soft_mask']}\n")
        f.write(f"  Mask Alpha: {cfg['mask_alpha']}\n")
        f.write(f"  Apply Gaussian: {cfg['apply_gaussian']}\n")
        if cfg['apply_gaussian']:
            f.write(f"  Gaussian Sigma: {cfg['gaussian_sigma']}\n")

    print("\n[AutoOpt] DONE. Files saved in /artifacts")
    print("  - optimized_gradcam_weights.json")
    print("  - gradcam_opt_params.json")
    print("  - gradcam_opt_report.txt")
    print("  - gradcam_weight_histogram.png")
