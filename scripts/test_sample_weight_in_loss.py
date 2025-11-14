"""
Test script to verify that sample_weight is passed to loss function.
"""
import tensorflow as tf
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.dataloader import get_binary_pipelines
from src.losses import create_simple_masked_loss
from src.model import build_model

# Load dataset
output_dir = project_root / "splitted_dataset"
train_ds, val_ds, test_ds, class_names = get_binary_pipelines(
    str(output_dir),
    img_size=(224, 224),
    batch_size=16,
    seed=42,
    use_masks=True,
    use_soft_mask=True,
    mask_alpha=0.2
)

# Check dataset format
print("Checking dataset format...")
batch = next(iter(train_ds))
print(f"Batch type: {type(batch)}")
print(f"Batch length: {len(batch)}")
if len(batch) == 3:
    x, y, w = batch
    print(f"✅ Dataset returns 3 elements: x={x.shape}, y={y.shape}, sample_weight={w.shape}")
    print(f"Sample weight mean: {tf.reduce_mean(w).numpy():.4f}")
else:
    print(f"❌ Dataset returns {len(batch)} elements (expected 3)")

# Create model and loss
model = build_model()
loss_fn = create_simple_masked_loss()

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=loss_fn,
    metrics=['accuracy']
)

# Test with a single batch
print("\nTesting loss function with sample_weight...")
x_batch, y_batch, w_batch = next(iter(train_ds.take(1)))

# Get predictions
y_pred = model(x_batch, training=False)

# Call loss function directly
print("\nCalling loss function directly...")
loss_value = loss_fn(y_batch, y_pred, sample_weight=w_batch)
print(f"Loss value: {loss_value.numpy():.4f}")

# Test without sample_weight
print("\nCalling loss function without sample_weight...")
loss_value_no_weight = loss_fn(y_batch, y_pred, sample_weight=None)
print(f"Loss value (no weight): {loss_value_no_weight.numpy():.4f}")

print("\n✅ Test completed!")