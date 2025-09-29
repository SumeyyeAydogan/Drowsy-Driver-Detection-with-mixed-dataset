import tensorflow as tf
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
def train_model(model, train_ds, val_ds, epochs=10, callbacks=None, initial_epoch=0):
    """
    Train the model with custom callbacks support
    """
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')]
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
    )
    
    '''
    train_small = train_ds.take(4)  # 4*16=64
    val_small = val_ds.take(4)
    history = model.fit(train_small, epochs=4, validation_data=val_small, class_weight=None)
    '''
    return history
