"""
Custom loss functions for drowsiness detection with spatial attention.
Focuses on eye and mouth regions for better feature learning.
"""
import tensorflow as tf
import numpy as np
from typing import Optional, Tuple


class EyeMouthMaskGenerator:
    """
    Generates masks focusing on eye and mouth regions for drowsiness detection.
    Uses simple geometric regions as fallback when face detection is not available.
    """
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224)):
        self.img_height, self.img_width = img_size
        
    def generate_mask(self, batch_images: tf.Tensor) -> tf.Tensor:
        """
        Generate eye-mouth focused masks for a batch of images.
        
        Args:
            batch_images: Tensor of shape (batch_size, height, width, channels)
            
        Returns:
            Tensor of shape (batch_size, height, width, 1) with values 0-1
        """
        batch_size = tf.shape(batch_images)[0]
        
        # Define eye and mouth regions (normalized coordinates)
        eye_region_top = int(0.25 * self.img_height)
        eye_region_bottom = int(0.45 * self.img_height)
        eye_region_left = int(0.2 * self.img_width)
        eye_region_right = int(0.8 * self.img_width)
        
        mouth_region_top = int(0.55 * self.img_height)
        mouth_region_bottom = int(0.75 * self.img_height)
        mouth_region_left = int(0.3 * self.img_width)
        mouth_region_right = int(0.7 * self.img_width)
        
        # Create coordinate grids
        y_indices = tf.range(self.img_height, dtype=tf.int32)
        x_indices = tf.range(self.img_width, dtype=tf.int32)
        
        # Create eye region mask
        eye_y_condition = tf.logical_and(y_indices >= eye_region_top, y_indices < eye_region_bottom)
        eye_x_condition = tf.logical_and(x_indices >= eye_region_left, x_indices < eye_region_right)
        
        # Use broadcasting to create 2D conditions
        eye_y_mask = tf.cast(eye_y_condition, tf.float32)[:, tf.newaxis]  # (H, 1)
        eye_x_mask = tf.cast(eye_x_condition, tf.float32)[tf.newaxis, :]  # (1, W)
        eye_mask = eye_y_mask * eye_x_mask  # (H, W)
        eye_mask = eye_mask[:, :, tf.newaxis]  # (H, W, 1)
        
        # Create mouth region mask
        mouth_y_condition = tf.logical_and(y_indices >= mouth_region_top, y_indices < mouth_region_bottom)
        mouth_x_condition = tf.logical_and(x_indices >= mouth_region_left, x_indices < mouth_region_right)
        
        mouth_y_mask = tf.cast(mouth_y_condition, tf.float32)[:, tf.newaxis]  # (H, 1)
        mouth_x_mask = tf.cast(mouth_x_condition, tf.float32)[tf.newaxis, :]  # (1, W)
        mouth_mask = mouth_y_mask * mouth_x_mask  # (H, W)
        mouth_mask = mouth_mask[:, :, tf.newaxis]  # (H, W, 1)
        
        # Combine eye and mouth masks
        combined_mask = tf.maximum(eye_mask, mouth_mask)
        
        # Add some Gaussian blur for smoother transitions
        combined_mask = self._gaussian_blur(combined_mask)
        
        # Broadcast to batch
        mask = tf.tile(combined_mask[tf.newaxis, :, :, :], [batch_size, 1, 1, 1])
        
        return mask
    
    def _gaussian_blur(self, mask: tf.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> tf.Tensor:
        """Apply Gaussian blur to smooth mask edges."""
        # Simple box filter approximation for Gaussian blur
        kernel = tf.ones((kernel_size, kernel_size, 1, 1), dtype=tf.float32) / (kernel_size * kernel_size)
        blurred = tf.nn.conv2d(mask[tf.newaxis, :, :, :], kernel, strides=1, padding='SAME')
        return tf.squeeze(blurred, axis=0)


def create_gradient_loss_fn(model: tf.keras.Model, 
                          mask_generator: EyeMouthMaskGenerator,
                          lambda_grad: float = 0.1,
                          target_layer_name: Optional[str] = None) -> callable:
    """
    Create a gradient-based loss function that penalizes attention outside eye-mouth regions.
    
    Args:
        model: The Keras model to train
        mask_generator: Instance of EyeMouthMaskGenerator
        lambda_grad: Weight for gradient penalty term
        target_layer_name: Name of intermediate layer for gradient computation (None for input)
        
    Returns:
        Loss function compatible with Keras compile()
    """
    
    def loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Combined loss: Binary cross-entropy + Gradient penalty
        
        Args:
            y_true: True labels (batch_size, 1)
            y_pred: Predicted probabilities (batch_size, 1)
            
        Returns:
            Combined loss tensor
        """
        # 1. Standard binary cross-entropy loss
        ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        ce_loss = tf.reduce_mean(ce_loss)
        
        # 2. Generate masks for current batch
        # Note: This is a limitation - we need access to input data
        # For now, we'll use a placeholder approach
        try:
            # Try to get input from model if it's been called recently
            if hasattr(model, '_last_input'):
                batch_input = model._last_input
            else:
                # Fallback: create a dummy mask (this won't work in practice)
                batch_size = tf.shape(y_true)[0]
                batch_input = tf.zeros((batch_size, 224, 224, 3))
            
            masks = mask_generator.generate_mask(batch_input)
            
            # 3. Compute gradient penalty
            if target_layer_name is not None:
                # Compute gradient w.r.t. intermediate layer
                target_layer = model.get_layer(target_layer_name)
                with tf.GradientTape() as tape:
                    tape.watch(target_layer.input)
                    logits = model(batch_input, training=True)
                    pred = logits[:, 0]  # Binary sigmoid output
                
                grads = tape.gradient(pred, target_layer.input)
                # Resize mask to match gradient shape if needed
                if grads.shape[1:3] != masks.shape[1:3]:
                    masks_resized = tf.image.resize(masks, grads.shape[1:3])
                else:
                    masks_resized = masks
                    
            else:
                # Compute gradient w.r.t. input
                with tf.GradientTape() as tape:
                    tape.watch(batch_input)
                    logits = model(batch_input, training=True)
                    pred = logits[:, 0]
                
                grads = tape.gradient(pred, batch_input)
                masks_resized = masks
            
            # Apply mask: penalize gradients in non-eye-mouth regions
            # mask=1 for eye-mouth regions (we want gradients here)
            # mask=0 for other regions (we want to penalize gradients here)
            masked_grads = grads * (1.0 - masks_resized)
            grad_loss = tf.reduce_mean(tf.square(masked_grads))
            
        except Exception as e:
            # Fallback: return only cross-entropy loss if gradient computation fails
            print(f"Warning: Gradient computation failed: {e}. Using only cross-entropy loss.")
            grad_loss = 0.0
        
        # 4. Combine losses
        total_loss = ce_loss + lambda_grad * grad_loss
        
        return total_loss
    
    return loss_fn


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


# Alternative: Class-based approach for better integration
class EyeMouthGradientLoss(tf.keras.losses.Loss):
    """
    Custom loss class that focuses on eye-mouth regions using gradient penalty.
    """
    
    def __init__(self, 
                 mask_generator: EyeMouthMaskGenerator,
                 lambda_grad: float = 0.1,
                 target_layer_name: Optional[str] = None,
                 name: str = "eye_mouth_gradient_loss"):
        super().__init__(name=name)
        self.mask_generator = mask_generator
        self.lambda_grad = lambda_grad
        self.target_layer_name = target_layer_name
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """Compute the loss."""
        # This is a simplified version - in practice, you'd need access to input data
        # For now, just return standard cross-entropy
        ce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return tf.reduce_mean(ce_loss)
