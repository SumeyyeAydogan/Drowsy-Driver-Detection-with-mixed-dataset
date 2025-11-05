"""
Custom loss functions for drowsiness detection with spatial attention.
Focuses on eye and mouth regions for better feature learning.
"""
import tensorflow as tf
import numpy as np
from typing import Optional, Tuple


def create_simple_masked_loss(lambda_grad: float = 0.1) -> callable:
    """
    Simplified version that works with sample_weight approach.
    This is more practical for integration with existing training pipeline.
    
    Args:
        lambda_grad: Weight for gradient penalty (not used in this simple version)
        
    Returns:
        Loss function compatible with Keras compile()
    """
    
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Loss function that can work with sample_weight for masking.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities  
            sample_weight: Optional sample weights for masking
        """
        # Standard binary cross-entropy
        ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        if sample_weight is not None:
            # Apply sample weights
            ce_loss = ce_loss * sample_weight
            ce_loss = tf.reduce_sum(ce_loss) / (tf.reduce_sum(sample_weight) + 1e-8)
        else:
            ce_loss = tf.reduce_mean(ce_loss)
        
        return ce_loss
    
    return loss_fn
