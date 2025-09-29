def save_model(model, path="model.h5"):
    """Save model to specified path"""
    model.save(path)
    print(f"✅ Model saved to: {path}")

def save_model_weights(model, path="model_weights.h5"):
    """Save only model weights to specified path"""
    model.save_weights(path)
    print(f"✅ Model weights saved to: {path}")