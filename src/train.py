import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from src.losses import create_simple_masked_loss, create_gradient_loss_fn, EyeMouthMaskGenerator, EyeMouthGradientLoss

def train_model(model, train_ds, val_ds, epochs=10, callbacks=None, initial_epoch=0, 
                use_gradient_loss=False, lambda_grad=0.1, target_layer_name=None):
    """
    Train the model with custom callbacks support and optional gradient-based loss.
    
    Args:
        model: Keras model to train
        train_ds: Training dataset
        val_ds: Validation dataset  
        epochs: Number of training epochs
        callbacks: List of Keras callbacks
        initial_epoch: Starting epoch number
        use_gradient_loss: Whether to use gradient-based loss for eye-mouth focus
        lambda_grad: Weight for gradient penalty term
        target_layer_name: Name of intermediate layer for gradient computation
    """
    
    # Choose loss function based on configuration
    if use_gradient_loss:
        # Use gradient-based loss (pixel-level masking)
        mask_generator = EyeMouthMaskGenerator()
        loss_fn = create_gradient_loss_fn(model, mask_generator, lambda_grad, target_layer_name)
    else:
        # Use standard loss with sample_weight support
        loss_fn = create_simple_masked_loss(lambda_grad)

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=loss_fn,
        metrics=['accuracy'],  # Keep basic accuracy unweighted
        weighted_metrics=[Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]  # These will use sample_weight
    )
    
    # Prepare callbacks
    if callbacks is None:
        callbacks = []
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1
        #sample_weight=True
        #sample_weight=train_ds.map(lambda x, y, w: w)
    )
    
    '''
    train_small = train_ds.take(4)  # 4*16=64
    val_small = val_ds.take(4)
    history = model.fit(train_small, epochs=4, validation_data=val_small, class_weight=None)
    '''
    return history
