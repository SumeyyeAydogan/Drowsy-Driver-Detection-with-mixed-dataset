import os
from datetime import datetime

from src.split_dataset   import split_dataset
from src.dataloader      import get_binary_pipelines
from src.model           import build_model
from src.train           import train_model
from src.utils           import plot_history, plot_metrics, create_run_directories, plot_dataset_distribution
from src.evaluate        import evaluate_model
from src.export          import save_model
from src.run_manager     import RunManager
from src.callbacks       import get_training_callbacks


if __name__ == "__main__":
    print("🚀 Starting Drowsy Driver Detection Project...")
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
    print(f"📁 Data directory: {output_dir}")

    # 3) Create Train/Val/Test folder hierarchy
    #    raw_dir contains => drowsy, notdrowsy
    split_dataset(raw_dir, output_dir, classes=("NotDrowsy", "Drowsy"), seed=42)

    # 4) Create run manager
    print("📁 Creating run manager...")
    run_manager = RunManager("20_epoch")
    print(f"✅ Run manager created: {run_manager.run_dir}")

    # 5) tf.data pipelines
    # LOADER (binary: NotDrowsy=0, Drowsy=1)
    print("🔄 Loading datasets (new binary pipeline)...")
    train_ds, val_ds, test_ds, class_names = get_binary_pipelines(
        output_dir,
        img_size=(224, 224),
        batch_size=32,
        seed=42
    )
    print("✅ Datasets loaded successfully!")

    # 5.1) Plot dataset distribution
    print("📊 Analyzing dataset distribution...")
    dist_plot_path = os.path.join(run_manager.run_dir, "plots", "dataset_distribution.png")
    # distribution plot
    plot_dataset_distribution(output_dir, save_path=dist_plot_path)
    print("✅ Dataset distribution analyzed and saved!")

    # 6) Build and train model
    print("🏗️  Building model...")
    model = build_model()
    print("✅ Model built successfully!")
    
    # 6.1) Check for existing checkpoint and load if available
    print("🔍 Checking for existing checkpoints...")
    initial_epoch = run_manager.load_latest_checkpoint(model)
    
    if initial_epoch > 0:
        print(f"🔄 Resuming training from epoch {initial_epoch + 1}")
    else:
        print("🆕 Starting training from scratch")
    
    # 7) Save initial config
    epoch_count=20
    config = {
        "run_name": run_manager.run_name,
        "epochs": epoch_count,
        "input_shape": (224, 224, 3),
        "model_type": "CNN",
        # OLD classes
        # "classes": ["notdrowsy", "drowsy"],
        # NEW classes from pipeline
        "classes": list(class_names),
        "batch_size": 32,
        "learning_rate": 1e-4,
        "started_at": str(datetime.now()),
        "initial_epoch": initial_epoch
    }
    run_manager.save_config(config)
    
    # 8) Training with all callbacks
    print("🎯 Starting training...")
    
    # Get all training callbacks (custom + standard Keras callbacks)
    callbacks = get_training_callbacks(run_manager)
    
    # Train the model
    history = train_model(
        model, 
        train_ds, 
        val_ds, 
        epochs=epoch_count,
        callbacks=callbacks,  # Add all callbacks
        initial_epoch=initial_epoch  # Resume from checkpoint if available
    )
    print("✅ Training completed!")

    # 9) Plot training graphs and save them
    print("📊 Plotting training history...")
    history_plot_path = os.path.join(run_manager.run_dir, "plots", "training_history.png")
    plot_history(history, save_path=history_plot_path)
    
    print("📈 Plotting metrics...")
    metrics_plot_path = os.path.join(run_manager.run_dir, "plots", "training_metrics.png")
    plot_metrics(history, save_path=metrics_plot_path)

    # 9.5) Evaluate on training set
    print("🧪 Evaluating model on training set...")
    train_plots_dir = os.path.join(run_manager.run_dir, "plots", "train")
    os.makedirs(train_plots_dir, exist_ok=True)
    evaluate_model(
        model,
        train_ds,
        plots_dir=train_plots_dir,
        subject_diverse_dir=os.path.join(output_dir, "train"),
        misclassified_only=True,
        ds_name="train"
    )
    print("✅ Training evaluation completed!")

    # 10) Evaluate on validation set
    print("🧪 Evaluating model on validation set...")
    evaluate_model(
        model,
        val_ds,
        plots_dir=os.path.join(run_manager.run_dir, "plots"),
        subject_diverse_dir=os.path.join(output_dir, "val"),
        ds_name="val"
    )
    #evaluate_validation(model, val_ds, plots_dir=os.path.join(run_manager.run_dir, "plots"))
    print("✅ Validation evaluation completed!")
    
    # 11) Evaluate on test set
    print("🧪 Evaluating model on test set...")
    evaluate_model(
        model,
        test_ds,
        plots_dir=os.path.join(run_manager.run_dir, "plots"),
        subject_diverse_dir=os.path.join(output_dir, "test"),
        ds_name="test"
    )
    print("✅ Test evaluation completed!")

    # 11) Save final model
    print("💾 Saving final model...")
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
    print(f"✅ Config saved to: {config_path}")
    
    print("\n" + "=" * 50)
    print("🎉 All tasks completed successfully!")
    print(f"📁 Results saved to: {run_manager.run_dir}")
    print("Project finished! 🚀")