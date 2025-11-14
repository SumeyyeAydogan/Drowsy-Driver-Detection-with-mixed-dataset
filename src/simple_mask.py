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
    
    def __init__(self, img_size: Tuple[int, int] = (224, 224), use_soft_mask: bool = False, alpha: float = 0.2):
        self.img_height, self.img_width = img_size
        
        # Pre-compute mask regions
        self._create_static_mask(use_soft_mask=use_soft_mask, alpha=alpha)
    
    def _create_static_mask(self, use_soft_mask: bool = False, alpha: float = 0.2):
        """Create static mask (using numpy).
        
        Args:
            use_soft_mask: If True, use soft masking (alpha blending) instead of hard mask
            alpha: Background transparency for soft masking (0.0 = fully masked, 1.0 = fully visible)
        """
        # Define regions
        eye_top = int(0.2 * self.img_height)
        eye_bottom = int(0.53 * self.img_height)
        eye_left = int(0.1 * self.img_width)
        eye_right = int(0.9 * self.img_width)
        
        mouth_top = int(0.57 * self.img_height)
        mouth_bottom = int(0.9 * self.img_height)
        mouth_left = int(0.2 * self.img_width)
        mouth_right = int(0.8 * self.img_width)
        
        if use_soft_mask:
            # Soft mask: ROI = 1.0, background = alpha
            mask = np.ones((self.img_height, self.img_width, 1), dtype=np.float32) * alpha
            
            # Eye region
            mask[eye_top:eye_bottom, eye_left:eye_right, 0] = 1.0
            
            # Mouth region
            mask[mouth_top:mouth_bottom, mouth_left:mouth_right, 0] = 1.0
        else:
            # Hard mask: ROI = 1.0, background = 0.0
            mask = np.zeros((self.img_height, self.img_width, 1), dtype=np.float32)
            
            # Eye region
            mask[eye_top:eye_bottom, eye_left:eye_right, 0] = 1.0
            
            # Mouth region
            mask[mouth_top:mouth_bottom, mouth_left:mouth_right, 0] = 1.0
        
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


def create_simple_mask_generator(img_size: Tuple[int, int] = (224, 224), use_soft_mask: bool = False, alpha: float = 0.2):
    """Factory function for simple mask generator.
    
    Args:
        img_size: Image dimensions (height, width)
        use_soft_mask: If True, use soft masking instead of hard mask
        alpha: Background transparency for soft masking (0.0 = fully masked, 1.0 = fully visible)
    """
    return SimpleEyeMouthMaskGenerator(img_size, use_soft_mask=use_soft_mask, alpha=alpha)
