# callbacks.py - Custom training callbacks
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
        
        print(f"✅ Epoch {epoch_num} completed and saved!")

def get_training_callbacks(run_manager):
    """
    Get all training callbacks including custom and standard Keras callbacks
    
    Args:
        run_manager: RunManager instance for custom checkpoint handling
        
    Returns:
        list: List of callbacks for training
    """
    return [
        CheckpointCallback(run_manager),
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_auc', factor=0.5, patience=3, min_lr=1e-6)
    ]
