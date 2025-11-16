import tensorflow as tf
from src.simple_mask import create_simple_mask_generator

def get_binary_pipelines(
    base_dir,
    img_size=(224, 224),
    batch_size=16,
    seed=42,
    class_names=("NotDrowsy", "Drowsy"),
    use_masks=False,
    use_soft_mask=False,
    mask_alpha=0.2,
    use_background_aug=False,  # Adversarial background augmentation
    bg_aug_prob=0.4,           # Probability of applying background augmentation
    bg_aug_face_ratio=0.4      # Face region ratio (center of image)
):
    AUTOTUNE = tf.data.AUTOTUNE

    # 1) Create datasets (binary labels)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/train",
        labels="inferred",
        label_mode="binary",
        class_names=list(class_names),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True,
        seed=seed
    )
    # Skip corrupted files/images during iteration (TF1.x/older TF2 compatibility)
    train_ds = train_ds.apply(tf.data.experimental.ignore_errors())

    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/val",
        labels="inferred",
        label_mode="binary",
        class_names=list(class_names),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False,
        seed=seed
    )
    val_ds = val_ds.apply(tf.data.experimental.ignore_errors())

    test_ds = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/test",
        labels="inferred",
        label_mode="binary",
        class_names=list(class_names),
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
        # seed not needed
    )
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())

    # 2) Augmentation (train only) + 3) Normalization
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])
    normalization = tf.keras.layers.Rescaling(1./255)

    # 4) Apply with map
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y),
        num_parallel_calls=AUTOTUNE
    )
    val_ds = val_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=AUTOTUNE
    )
    test_ds = test_ds.map(
        lambda x, y: (normalization(x), y),
        num_parallel_calls=AUTOTUNE
    )

    # 4.5) Add adversarial background augmentation (BEFORE masks, AFTER normalization)
    if use_background_aug:
        from src.adversarial_augmentation import RandomBackgroundReplacement
        bg_aug_layer = RandomBackgroundReplacement(
            prob=bg_aug_prob,
            face_center_ratio=bg_aug_face_ratio
        )
        # Apply only to training set
        train_ds = train_ds.map(
            lambda x, y: (bg_aug_layer(x, training=True), y),
            num_parallel_calls=AUTOTUNE
        )

    # 5) Add masks if requested
    if use_masks:
        mask_generator = create_simple_mask_generator(img_size, use_soft_mask=use_soft_mask, alpha=mask_alpha)
        
        def add_dynamic_weights(x, y):
            masks = mask_generator.generate_mask(x)  # (batch, H, W, 1)
            # Dynamic per-sample weight: ROI intensity relative to global intensity
            eps = tf.constant(1e-8, dtype=x.dtype)
            roi_intensity = tf.reduce_sum(x * masks, axis=[1, 2, 3])
            total_intensity = tf.reduce_sum(x, axis=[1, 2, 3]) + eps
            focus_ratio = roi_intensity / total_intensity  # (batch,)

            # Normalize to mean ≈ 1 within batch for stability
            batch_mean = tf.reduce_mean(focus_ratio) + eps
            sample_weights = focus_ratio / batch_mean
            
            # Optional: Clip weights only if they exceed reasonable bounds
            # Current observed range: ~0.63-1.34, so clipping at 0.5-1.5 has minimal effect
            # Uncomment below if you want to enforce stricter bounds (e.g., 0.7-1.3)
            # min_weight = tf.constant(0.7, dtype=sample_weights.dtype)
            # max_weight = tf.constant(1.3, dtype=sample_weights.dtype)
            # sample_weights = tf.clip_by_value(sample_weights, min_weight, max_weight)
            # batch_mean_after_clip = tf.reduce_mean(sample_weights) + eps
            # sample_weights = sample_weights / batch_mean_after_clip
            
            # Ensure minimum value to avoid numerical issues (very small threshold)
            sample_weights = tf.maximum(sample_weights, tf.constant(1e-4, dtype=sample_weights.dtype))
            
            return x, y, sample_weights
        
        train_ds = train_ds.map(add_dynamic_weights, num_parallel_calls=AUTOTUNE)
        # Note: val_ds and test_ds don't use sample_weight for evaluation
        # This ensures validation/test metrics reflect real-world performance
        # where sample_weight won't be available

    # 6) Performance: cache + prefetch
    # (Extra shuffling on train helps)
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    # class_names is stored on Dataset objects; still returning for convenience.
    return train_ds, val_ds, test_ds, list(class_names)