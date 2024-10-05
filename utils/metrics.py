from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, precision_recall_curve
import numpy as np


def calculate_metrics(y_true, y_pred, y_score):
    n_classes = y_true.shape[1]

    # Per-disease metrics
    auc_roc = []
    ap = []
    f1 = []
    specificity = []
    sensitivity = []
    auprc = []
    cm = []

    for i in range(n_classes):
        auc_roc.append(roc_auc_score(y_true[:, i], y_score[:, i]))
        ap.append(average_precision_score(y_true[:, i], y_score[:, i]))
        f1.append(f1_score(y_true[:, i], y_pred[:, i]))
        tn, fp, fn, tp = confusion_matrix(y_true[:, i], y_pred[:, i]).ravel()
        specificity.append(tn / (tn + fp))
        sensitivity.append(tp / (tp + fn))
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        auprc.append(np.trapz(precision, recall))
        cm.append(confusion_matrix(y_true[:, i], y_pred[:, i]))

    # Overall metrics
    micro_auc_roc = roc_auc_score(y_true, y_score, average='micro')
    macro_auc_roc = roc_auc_score(y_true, y_score, average='macro')
    micro_ap = average_precision_score(y_true, y_score, average='micro')
    macro_ap = average_precision_score(y_true, y_score, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')

    return {
        'overall': {
            'micro_auc_roc': micro_auc_roc,
            'macro_auc_roc': macro_auc_roc,
            'micro_ap': micro_ap,
            'macro_ap': macro_ap,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
        },
        'per_disease': {
            'auc_roc': auc_roc,
            'ap': ap,
            'f1': f1,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'auprc': auprc,
            'cm': cm,
        }
    }

