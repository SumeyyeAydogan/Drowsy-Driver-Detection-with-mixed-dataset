"""
Simple mask generation alternative implementation.
Compatible with TensorFlow graph mode, less complex.
"""
import tensorflow as tf
import numpy as np
from typing import Tuple


class SimpleEyeMouthMaskGenerator:
    """
    Simple eye-mouth mask generator.
    Fully compatible with TensorFlow graph mode.
    """
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224)):
        self.img_height, self.img_width = img_size
        
        # Pre-compute mask regions
        self._create_static_mask()
    
    def _create_static_mask(self):
        """Create static mask (using numpy)."""
        # Define regions
        eye_top = int(0.25 * self.img_height)
        eye_bottom = int(0.45 * self.img_height)
        eye_left = int(0.2 * self.img_width)
        eye_right = int(0.8 * self.img_width)
        
        mouth_top = int(0.55 * self.img_height)
        mouth_bottom = int(0.75 * self.img_height)
        mouth_left = int(0.3 * self.img_width)
        mouth_right = int(0.7 * self.img_width)
        
        # Create mask
        mask = np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)
        
        # Eye region
        mask[eye_top:eye_bottom, eye_left:eye_right, 0] = 1.0
        
        # Mouth region
        mask[mouth_top:mouth_bottom, mouth_left:mouth_right, 0] = 1.0
        
        # No blur for now - keep it simple
        # Can be added later if needed
        
        # Convert to tensor
        self.static_mask = tf.constant(mask, dtype=tf.float32)
    
    def generate_mask(self, batch_images: tf.Tensor) -> tf.Tensor:
        """
        Generate masks for a batch of images.
        
        Args:
            batch_images: Tensor of shape (batch_size, height, width, channels)
            
        Returns:
            Tensor of shape (batch_size, height, width, 1) with values 0-1
        """
        batch_size = tf.shape(batch_images)[0]
        
        # Broadcast static mask to batch
        mask = tf.tile(self.static_mask[tf.newaxis, :, :, :], [batch_size, 1, 1, 1])
        
        return mask


def create_simple_mask_generator(img_size: Tuple[int, int] = (224, 224)):
    """Factory function for simple mask generator."""
    return SimpleEyeMouthMaskGenerator(img_size)
