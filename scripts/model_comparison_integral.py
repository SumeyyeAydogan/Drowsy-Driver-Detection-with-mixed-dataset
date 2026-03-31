import os, sys, json
import numpy as np
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image, ImageDraw
import mediapipe as mp

# Add root path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gradcam import CustomGradCAM


# ================== CONFIG ======================
CONFIG = {
    "data_dir": r"splitted_dataset/test",
    "img_size": (224, 224),
    "dataset_name": "test",
    "landmark_box_half_size": 12,
    "background_mask_value": 0.2,   # mask background (soft)
    "threshold_source": "baseline_median",  # baseline median as threshold
    "hist_bins": 50,
}

# IMPORTANT: "orijinal" must exist for threshold
MODEL_CONFIGS = [
    {"label": "orijinal",   "model_path": r"runs/30_epoch_baseline/models/final_model.h5"},
    {"label": "reward",     "model_path": r"runs/30_epoch_reward-landmark-soft/models/final_model.h5"},
    {"label": "exp-reward", "model_path": r"runs/30_epoch_exp-reward-landmark-soft/models/final_model.h5"},
]

# ================== MediaPipe FaceMesh ======================
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)

LEFT_EYE_IDX  = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362]
MOUTH_IDX     = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
ROI_IDX = LEFT_EYE_IDX + RIGHT_EYE_IDX + MOUTH_IDX


# ================== MASK + FOCUS ======================
def create_landmark_mask(image_np_uint8, img_size):
    """
    image_np_uint8: (H,W,3) RGB uint8 at img_size
    returns: (H,W) float32 mask in [0,1] or None if no face
    """
    h, w = img_size
    background_value = float(CONFIG.get("background_mask_value", 0.0))

    results = mp_face_mesh.process(image_np_uint8)
    if not results.multi_face_landmarks:
        return None

    face = results.multi_face_landmarks[0]

    bg_pil_value = int(background_value * 255)
    pil_mask = Image.new("L", (w, h), bg_pil_value)
    draw = ImageDraw.Draw(pil_mask)

    box_half_size = int(CONFIG["landmark_box_half_size"])
    for i in ROI_IDX:
        lm = face.landmark[i]
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        x0 = max(0, cx - box_half_size)
        y0 = max(0, cy - box_half_size)
        x1 = min(w - 1, cx + box_half_size)
        y1 = min(h - 1, cy + box_half_size)
        draw.rectangle([x0, y0, x1, y1], outline=255, fill=255)

    mask = np.array(pil_mask, dtype=np.float32) / 255.0
    return mask


def compute_focus_ratio(heatmap, mask):
    heatmap = np.maximum(heatmap, 0)
    mx = float(heatmap.max())
    if mx > 0:
        heatmap = heatmap / (mx + 1e-8)
    focus = float(np.sum(heatmap * mask))
    total = float(np.sum(heatmap) + 1e-8)
    return focus / total


# ================== CORE ======================
def collect_focus_ratios(model, data_dir, img_size):
    """
    Returns:
      focus_ratios: np.array (only face_ok==1)
      stats: dict with N_total, N_face, face_rate
    """
    gradcam = CustomGradCAM(model)

    ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=["NotDrowsy", "Drowsy"],  # NotDrowsy=0, Drowsy=1
        image_size=img_size,
        batch_size=1,
        shuffle=False
    )
    file_paths = list(getattr(ds, "file_paths", []))
    path_ds = tf.data.Dataset.from_tensor_slices(file_paths).batch(1)
    ds = tf.data.Dataset.zip((ds, path_ds))
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ratios = []
    face_ok = 0
    total = 0

    for idx, (data_batch, path_batch) in enumerate(ds):
        total += 1
        images, labels = data_batch
        image = images[0].numpy()

        # For MediaPipe: uint8 0..255
        image_uint8 = (image * 255.0).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        # For model: float 0..1
        image_norm = image_uint8 / 255.0

        # prediction -> class_idx for gradcam
        prob = float(model.predict(image_norm[None, ...], verbose=0)[0][0])
        pred = 1 if prob >= 0.5 else 0

        # GradCAM heatmap (pred class)
        heatmap = gradcam.compute_heatmap(image_norm, class_idx=pred)
        heatmap = tf.image.resize(
            heatmap[..., None],
            img_size,
            method="bilinear",
            antialias=True
        ).numpy()[..., 0]

        mask = create_landmark_mask(image_uint8, img_size)
        if mask is None:
            pass
        else:
            face_ok += 1
            ratios.append(compute_focus_ratio(heatmap, mask))

        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx+1}/{len(file_paths)} | face_ok={face_ok}")

    ratios = np.array(ratios, dtype=np.float32)
    stats = {
        "N_total": int(total),
        "N_face": int(face_ok),
        "face_rate": float(face_ok / max(total, 1))
    }
    return ratios, stats


# ================== PLOT ======================
def plot_histograms(results_dict, dataset_name, out_path):
    """
    results_dict: label -> np.array focus_ratios
    """
    labels = [k for k in ["orijinal", "reward", "exp-reward"] if k in results_dict]
    if not labels:
        print("[WARN] No results to plot.")
        return

    # common bins
    all_vals = np.concatenate([results_dict[l] for l in labels if len(results_dict[l]) > 0])
    if len(all_vals) == 0:
        print("[WARN] Empty arrays, skip plot.")
        return

    bins = int(CONFIG.get("hist_bins", 50))
    x_min, x_max = float(all_vals.min()), float(all_vals.max())
    bin_edges = np.linspace(x_min, x_max, bins + 1)

    fig, axes = plt.subplots(1, len(labels), figsize=(5 * len(labels), 4), squeeze=False)
    fig.suptitle(f"Focus Ratio Distributions - {dataset_name}", fontsize=14, fontweight="bold")

    for i, label in enumerate(labels):
        ax = axes[0, i]
        vals = results_dict[label]
        ax.hist(vals, bins=bin_edges, density=True, edgecolor="black", alpha=0.75)
        ax.set_title(f"{label} (n={len(vals)})")
        ax.set_xlabel("Focus Ratio")
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)

        ax.axvline(np.mean(vals), linestyle="--", linewidth=2, label=f"Mean {np.mean(vals):.3f}")
        ax.axvline(np.median(vals), linestyle="--", linewidth=2, label=f"Median {np.median(vals):.3f}")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {out_path}")


# ================== SUMMARY SAVE ======================
def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ================== MAIN ======================
if __name__ == "__main__":
    cfg = CONFIG
    data_dir = cfg["data_dir"]
    img_size = tuple(cfg["img_size"])
    dataset_name = cfg.get("dataset_name", Path(data_dir).name)

    os.makedirs("artifacts", exist_ok=True)
    summary_path = os.path.join("artifacts", f"focus_summary_{dataset_name}.jsonl")

    # 1) Collect ratios per model
    ratios_by_model = {}
    stats_by_model = {}
    model_paths = {}

    for m_cfg in MODEL_CONFIGS:
        label = m_cfg["label"]
        model_path = m_cfg["model_path"]
        model_paths[label] = model_path

        print(f"\n[LOAD] {label}: {model_path}")
        if not os.path.exists(model_path):
            print(f"[WARN] Missing model path, skip: {model_path}")
            continue

        model = tf.keras.models.load_model(model_path, compile=False)

        print(f"[RUN] Collecting focus ratios for {label} ...")
        ratios, stats = collect_focus_ratios(model, data_dir, img_size)

        ratios_by_model[label] = ratios
        stats_by_model[label] = stats

        print(f"[DONE] {label}: N_face={stats['N_face']} / N_total={stats['N_total']} | mean={ratios.mean():.4f} | median={np.median(ratios):.4f}")

    if "orijinal" not in ratios_by_model or len(ratios_by_model["orijinal"]) == 0:
        raise RuntimeError("Baseline 'orijinal' ratios missing/empty. Cannot compute threshold.")

    # 2) Threshold from baseline (median)
    T = float(np.median(ratios_by_model["orijinal"]))
    print("\n================= THRESHOLD =================")
    print(f"Threshold source: baseline median")
    print(f"T = median(focus_orijinal) = {T:.4f}")

    # 3) Compute integral metric and write summaries
    print("\n================= INTEGRAL METRIC =================")
    # "Integral" == P(focus > T)
    # Baseline P(focus > T)
    baseline_ratios = ratios_by_model["orijinal"]
    P_baseline = float(np.mean(baseline_ratios > T))

    for label, ratios in ratios_by_model.items():
        if len(ratios) == 0:
            continue

        p_above = float(np.mean(ratios > T))
        delta_p = p_above - P_baseline   # <<< YENİ METRİK

        mean_v = float(np.mean(ratios))
        med_v = float(np.median(ratios))
        std_v = float(np.std(ratios))

        print(
            f"{label:10s} | "
            f"P(focus>T)={p_above:.4f} | "
            f"ΔP={delta_p:+.4f} | "
            f"mean={mean_v:.4f} | "
            f"median={med_v:.4f}"
        )

        summary = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset": dataset_name,
            "data_dir": data_dir,
            "img_size": list(img_size),
            "model_label": label,
            "model_path": model_paths.get(label, ""),
            "threshold_source": cfg.get("threshold_source", "baseline_median"),
            "threshold_T": T,

            # CORE METRICS
            "P_focus_above_T": p_above,
            "delta_P_vs_baseline": delta_p,   # <<< KAYITLI

            "mean_focus": mean_v,
            "median_focus": med_v,
            "std_focus": std_v,

            "N_total": stats_by_model[label]["N_total"],
            "N_face": stats_by_model[label]["N_face"],
            "face_rate": stats_by_model[label]["face_rate"],
            "mask": {
                "landmark_box_half_size": cfg["landmark_box_half_size"],
                "background_mask_value": cfg["background_mask_value"],
            }
        }

        append_jsonl(summary_path, summary)

    print(f"\n[SAVE] Summary appended to: {summary_path}")

    # 4) Plot histograms (optional but useful)
    hist_path = os.path.join("artifacts", f"model_focus_comparison_mask_{dataset_name}.png")
    plot_histograms(ratios_by_model, dataset_name, hist_path)

    print("\n[DONE]")
