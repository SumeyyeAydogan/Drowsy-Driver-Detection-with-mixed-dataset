import numpy as np
from sklearn.metrics import classification_report
from src.utils import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from src.gradcam import analyze_model_gradcam
import os

def evaluate_model(
    model,
    test_ds,
    plots_dir=None,
    class_names=['Not Drowsy', 'Drowsy'],
    subject_diverse_dir=None,
    misclassified_only=False,
    ds_name="test",
    num_gradcam_samples=10,
    confusion=True
):
    """
    Evaluate model performance on test dataset
    """
    # 1) Collect all test predictions and labels
    y_true = []
    y_pred = []
    y_pred_proba = []
    
    for x_batch, y_batch in test_ds:
        # Get predictions
        preds = model.predict(x_batch, verbose=0)
        
        # For binary classification: y_batch is already 0 or 1
        y_true.extend(y_batch.numpy().flatten())
        y_pred.extend((preds > 0.5).astype(int).flatten())  # Threshold 0.5
        y_pred_proba.extend(preds.flatten())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_proba = np.array(y_pred_proba)
    
    # 2) Print classification report
    print("Classification Report:")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 3) Plot confusion matrix
    print("Plotting Confusion Matrix...")
    cm_save_path = os.path.join(plots_dir, f"{ds_name}_confusion_matrix.png") if plots_dir else None
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_save_path)
    
    # 4) Plot ROC curve
    print("Plotting ROC Curve...")
    roc_save_path = os.path.join(plots_dir, f"{ds_name}_roc_curve.png") if plots_dir else None
    plot_roc_curve(y_true, y_pred_proba, save_path=roc_save_path)
    
    # 5) Plot Precision-Recall curve
    print("Plotting Precision-Recall Curve...")
    pr_save_path = os.path.join(plots_dir, f"{ds_name}_precision_recall_curve.png") if plots_dir else None
    plot_precision_recall_curve(y_true, y_pred_proba, save_path=pr_save_path)
    
    # 6) Generate GradCAM visualizations for explainability
    print("Generating GradCAM visualizations...")
    gradcam_dir = os.path.join(plots_dir, f"{ds_name}_gradcam") if plots_dir else f"{ds_name}_gradcam_results"
    os.makedirs(gradcam_dir, exist_ok=True)
    analyze_model_gradcam(
        model,
        test_ds,
        num_samples=num_gradcam_samples,
        output_dir=gradcam_dir,
        class_names=tuple(class_names),
        subject_diverse_dir=subject_diverse_dir,
        misclassified_only=misclassified_only,
        confusion=confusion
    )
    
    # 7) Return metrics for further analysis
    return {
        'y_true': y_true,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

