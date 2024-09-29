import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, \
    label_ranking_average_precision_score


def calculate_metrics(y_true, y_pred, y_score):
    n_classes = y_true.shape[1]

    # Per-class metrics
    auc_roc = [roc_auc_score(y_true[:, i], y_score[:, i]) for i in range(n_classes)]
    ap = [average_precision_score(y_true[:, i], y_score[:, i]) for i in range(n_classes)]
    f1 = [f1_score(y_true[:, i], y_pred[:, i]) for i in range(n_classes)]

    # Overall metrics
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    mean_ap = np.mean(ap)
    subset_accuracy = (y_true == y_pred).all(axis=1).mean()

    # Label Ranking Average Precision
    lrap = label_ranking_average_precision_score(y_true, y_score)

    # Confusion matrices
    cm_per_class = [confusion_matrix(y_true[:, i], y_pred[:, i]) for i in range(n_classes)]

    return {
        'auc_roc': auc_roc,
        'ap': ap,
        'f1': f1,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'mean_ap': mean_ap,
        'subset_accuracy': subset_accuracy,
        'lrap': lrap,
        'cm_per_class': cm_per_class
    }

