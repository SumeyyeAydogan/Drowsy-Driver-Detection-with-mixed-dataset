import os
from datetime import datetime
import tensorflow as tf

from src.split_dataset   import split_dataset
from src.dataloader      import get_binary_pipelines
from src.model           import build_model
from src.train           import train_model
from src.utils           import plot_history, plot_metrics, create_run_directories, plot_dataset_distribution
from src.evaluate        import evaluate_model
from src.gradcam_analysis import analyze_tf_keras_gradcam
from src.run_manager     import RunManager
from src.callbacks       import get_training_callbacks
import splitfolders

if __name__ == "__main__":
    print("?? Starting Drowsy Driver Detection Project...")
    print("=" * 50)
    
    # Project root directory: the folder where this file is located
    import os
    project_root = os.path.dirname(os.path.abspath(__file__))
    #project_root = r"D:\internship\Drowsy-Driver-Detection-Project"

    # 1) Raw data folder (what you have)
    raw_dir = os.path.join(project_root, "dataset")
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"`dataset` not found: {raw_dir}")

    # 2) Folder where split data will go
    output_dir = os.path.join(project_root, "splitted_dataset")
    print(f"?? Data directory: {output_dir}")

    # 3) Create Train/Val/Test folder hierarchy
    #    raw_dir contains => drowsy, notdrowsy
    #split_dataset(raw_dir, output_dir, classes=("NotDrowsy", "Drowsy"), seed=42)
    #splitfolders.ratio(raw_dir, output=output_dir, seed=1337, ratio=(.8, 0.15, 0.05))
    
    # 4) Create run manager
    print("?? Creating run manager...")
    run_manager = RunManager("30_epoch_with-mask_sbj-gradcam")
    print(f"? Run manager created: {run_manager.run_dir}")

    # 5) tf.data pipelines
    # LOADER (binary: NotDrowsy=0, Drowsy=1)
    print("?? Loading datasets...")
    train_ds, val_ds, test_ds, class_names = get_binary_pipelines(
        output_dir,
        img_size=(224, 224),
        batch_size=32,
        seed=42,
        use_masks=True  # Enable eye-mouth focused masks
    )
    print("? Datasets loaded successfully!")
    '''
    # Debug dataset shapes (support (x,y) and (x,y,w))
    def _print_batch_info(ds, name):
        for batch in ds.take(2):
            if isinstance(batch, (tuple, list)) and len(batch) == 3:
                x_batch, y_batch, w_batch = batch
                print(f"{name} x:", x_batch.shape, " y:", y_batch.shape, " w:", w_batch.shape)
                print(tf.reduce_mean(y_batch), tf.reduce_mean(w_batch))
            else:
                x_batch, y_batch = batch
                print(f"{name} x:", x_batch.shape, " y:", y_batch.shape)
                print(tf.reduce_mean(y_batch))

    _print_batch_info(train_ds, "train")
    _print_batch_info(val_ds, "val")
    _print_batch_info(test_ds, "test")
    '''
    # 5.1) Plot dataset distribution
    print("?? Analyzing dataset distribution...")
    dist_plot_path = os.path.join(run_manager.run_dir, "plots", "dataset_distribution.png")
    # distribution plot
    plot_dataset_distribution(output_dir, save_path=dist_plot_path)
    print("? Dataset distribution analyzed and saved!")

    # 6) Build and train model
    print("???  Building model...")
    model = build_model()
    print("? Model built successfully!")
    
    # 6.1) Check for existing checkpoint and load if available
    print("?? Checking for existing checkpoints...")
    initial_epoch = run_manager.load_latest_checkpoint(model)
    
    if initial_epoch > 0:
        print(f"?? Resuming training from epoch {initial_epoch + 1}")
    else:
        print("?? Starting training from scratch")
    
    # 7) Save initial config
    epoch_count=30
    config = {
        "run_name": run_manager.run_name,
        "epochs": epoch_count,
        "input_shape": (224, 224, 3),
        "model_type": "CNN",
        "classes": list(class_names),
        "batch_size": 32,
        "learning_rate": 1e-4,
        "started_at": str(datetime.now()),
        "initial_epoch": initial_epoch
    }
    run_manager.save_config(config)
    
    # 8) Training with all callbacks
    print("?? Starting training...")
    
    # Get all training callbacks (custom + standard Keras callbacks)
    gradcam_epoch_outputs = os.path.join(run_manager.run_dir, "gradcam_epoch_outputs")
    gradcam_log_file = os.path.join(run_manager.run_dir, "gradcam_debug.log")
    callbacks = get_training_callbacks(run_manager, val_ds, gradcam_epoch_outputs, 
                                      max_samples=10, gradcam_log_file=gradcam_log_file)
    
    # Train the model
    history = train_model(
        model, 
        train_ds, 
        val_ds, 
        epochs=epoch_count,
        callbacks=callbacks,  # Add all callbacks
        initial_epoch=initial_epoch,  # Resume from checkpoint if available
        #use_gradient_loss=False,  # Use sample_weight approach for better integration
        lambda_grad=0.1,  # Weight for gradient penalty
        #target_layer_name="conv2d_2"  # Target intermediate layer for gradient computation
    )
    print("? Training completed!")

    # 9) Plot training graphs and save them
    print("?? Plotting training history...")
    history_plot_path = os.path.join(run_manager.run_dir, "plots", "training_history.png")
    plot_history(history, save_path=history_plot_path)
    
    print("?? Plotting metrics...")
    metrics_plot_path = os.path.join(run_manager.run_dir, "plots", "training_metrics.png")
    plot_metrics(history, save_path=metrics_plot_path)

    
    # 9.5) Evaluate on training set
    print("?? Evaluating model on training set...")
    train_plots_dir = os.path.join(run_manager.run_dir, "plots", "train_gradcam")
    os.makedirs(train_plots_dir, exist_ok=True)
    analyze_tf_keras_gradcam(
        model=model,
        test_ds=train_ds,
        output_dir=train_plots_dir,
        num_samples=30,
        class_names=tuple(class_names)
    )
    print("? Training evaluation completed!")

    # 10) Evaluate on validation set
    print("?? Evaluating model on validation set...")
    evaluate_model(
        model,
        val_ds,
        plots_dir=os.path.join(run_manager.run_dir, "plots"),
        subject_diverse_dir=os.path.join(output_dir, "val"),
        ds_name="val",
    )
    print("? Validation evaluation completed!")
    
    # 11) Evaluate on test set
    print("?? Evaluating model on test set...")
    evaluate_model(
        model,
        test_ds,
        plots_dir=os.path.join(run_manager.run_dir, "plots"),
        subject_diverse_dir=os.path.join(output_dir, "test"),
        ds_name="test",
    )
    print("? Test evaluation completed!")

    # 11) Save final model
    print("?? Saving final model...")
    run_manager.save_final_model(model)
    
    # 12) Save simple config
    config = {
        "run_name": run_manager.run_name,
        "epochs": epoch_count,
        "input_shape": (224, 224, 3),
        "model_type": "CNN",
        "classes": list(class_names)
    }
    
    import json
    config_path = os.path.join(run_manager.run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"? Config saved to: {config_path}")
    
    print("\n" + "=" * 50)
    print("?? All tasks completed successfully!")
    print(f"?? Results saved to: {run_manager.run_dir}")
    print("Project finished! ??")