import tensorflow as tf

def get_binary_pipelines(
    base_dir,
    img_size=(224, 224),
    batch_size=32,
    seed=42,
    class_names=("NotDrowsy", "Drowsy")  # 0->NotDrowsy, 1->Drowsy - FIXED ORDER
):
    AUTOTUNE = tf.data.AUTOTUNE

    # 1) Create datasets (binary labels)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{base_dir}/train",
        labels="inferred",
        label_mode="binary",            # <— BINARY
        class_names=list(class_names),  # <— Fixed class order
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

    # 5) Performance: cache + prefetch
    # (Extra shuffling on train helps)
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(AUTOTUNE)
    val_ds   = val_ds.cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(AUTOTUNE)

    # class_names is stored on Dataset objects; still returning for convenience.
    return train_ds, val_ds, test_ds, list(class_names)