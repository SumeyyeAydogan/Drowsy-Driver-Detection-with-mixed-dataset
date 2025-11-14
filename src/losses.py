"""
Custom loss functions for drowsiness detection with spatial attention.
Focuses on eye and mouth regions for better feature learning.
"""
import tensorflow as tf
from typing import Optional


class SimpleMaskedLoss(tf.keras.losses.Loss):
    """
    Loss class that properly handles sample_weight for masking.
    Inheriting from tf.keras.losses.Loss ensures sample_weight is passed correctly.
    """
    
    def __init__(self, name='simple_masked_loss', **kwargs):
        super().__init__(name=name, **kwargs)
    
    def __call__(self, y_true, y_pred, sample_weight=None, **kwargs):
        """
        Override __call__ to properly handle sample_weight without base class squeeze.
        This is called by Keras during training.
        """
        # Convert to tensors if needed
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        
        # Call our custom call method directly, bypassing base class __call__
        loss = self.call(y_true, y_pred, sample_weight=sample_weight)
        
        # Apply reduction if needed (but we handle it in call method)
        return loss
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor, sample_weight: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Loss function that can work with sample_weight for masking.
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities  
            sample_weight: Optional sample weights for masking
        """
        # Standard binary cross-entropy
        ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        
        # Debug: Log sample_weight status (use tf.print which works in graph mode)
        if sample_weight is not None:
            tf.print("[MaskedLoss] ✅ sample_weight RECEIVED | shape:", tf.shape(sample_weight),
                    "| mean:", tf.reduce_mean(sample_weight),
                    "| min:", tf.reduce_min(sample_weight),
                    "| max:", tf.reduce_max(sample_weight),
                    "| y_true shape:", tf.shape(y_true),
                    "| y_pred shape:", tf.shape(y_pred))
            sample_weight = tf.cast(sample_weight, ce_loss.dtype)
            sample_weight = tf.reshape(sample_weight, tf.shape(ce_loss))

            with tf.control_dependencies([
                tf.debugging.assert_equal(
                    tf.shape(sample_weight),
                    tf.shape(ce_loss),
                    message="[MaskedLoss] sample_weight shape does not match ce_loss"
                ),
                tf.debugging.assert_greater_equal(
                    sample_weight,
                    tf.zeros_like(sample_weight),
                    message="[MaskedLoss] sample_weight contains negative values"
                )
            ]):
                total_weight = tf.reduce_sum(sample_weight)

            tf.debugging.assert_positive(
                total_weight + 1e-8,
                message="[MaskedLoss] Sum of sample_weight must be > 0"
            )

            ce_loss = ce_loss * sample_weight
            ce_loss = tf.reduce_sum(ce_loss) / (total_weight + 1e-8)
        else:
            tf.print("[MaskedLoss] ⚠️  sample_weight is None | y_true shape:", tf.shape(y_true),
                    "| y_pred shape:", tf.shape(y_pred),
                    "| Falling back to uniform weighting.")
            ce_loss = tf.reduce_mean(ce_loss)
        
        return ce_loss


def create_simple_masked_loss() -> callable:
    """
    Factory function that returns a loss instance compatible with Keras compile().
    This ensures sample_weight is properly passed to the loss function.
        
    Returns:
        Loss instance compatible with Keras compile()
    """
    return SimpleMaskedLoss()
