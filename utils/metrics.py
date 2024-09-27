import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, confusion_matrix

def calculate_metrics(y_pred, y_true):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Calculate mAP
    mAP = average_precision_score(y_true, y_pred, average='macro')

    # Calculate AUC-ROC
    auc_roc = roc_auc_score(y_true, y_pred, average='macro')

    # Calculate F1-score
    y_pred_binary = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_binary, average='macro')

    # Calculate Specificity and Sensitivity
    cm = confusion_matrix(y_true.ravel(), y_pred_binary.ravel())
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    # Calculate per-class metrics
    per_class_metrics = {}
    for i in range(y_true.shape[1]):
        per_class_metrics[f'class_{i}_ap'] = average_precision_score(y_true[:, i], y_pred[:, i])
        per_class_metrics[f'class_{i}_auc'] = roc_auc_score(y_true[:, i], y_pred[:, i])
        per_class_metrics[f'class_{i}_f1'] = f1_score(y_true[:, i], y_pred_binary[:, i])

    metrics = {
        'mAP': mAP,
        'AUC-ROC': auc_roc,
        'F1-score': f1,
        'Specificity': specificity,
        'Sensitivity': sensitivity,
        **per_class_metrics
    }

    return metrics

