import torch
import numpy as np

def calculate_metrics(preds, targets, num_classes=6, smooth=1e-6):
    """
    Computes precision, recall, f1, and IoU for 3D tensors.
    Inputs are expected as (Batch, D, H, W).
    """
    metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'iou': []
    }
    
    # Iterate through classes (excluding background if desired, but 
    # here we include it as per standard practice unless specified)
    for cls in range(num_classes):
        pred_cls = (preds == cls).float()
        target_cls = (targets == cls).float()
        
        # Pixel-wise intersection, union, etc.
        intersection = (pred_cls * target_cls).sum()
        total_pred = pred_cls.sum()
        total_target = target_cls.sum()
        union = total_pred + total_target - intersection
        
        # Precision
        precision = (intersection + smooth) / (total_pred + smooth)
        # Recall
        recall = (intersection + smooth) / (total_target + smooth)
        # F1 Score
        f1 = (2 * intersection + smooth) / (total_pred + total_target + smooth)
        # IoU
        iou = (intersection + smooth) / (union + smooth)
        
        metrics['precision'].append(precision.item())
        metrics['recall'].append(recall.item())
        metrics['f1'].append(f1.item())
        metrics['iou'].append(iou.item())
        
    return metrics

def confusion_matrix(preds, targets, num_classes=6):
    """
    Computes the (num_classes, num_classes) confusion matrix for 3D volumes.

    Args:
        preds, targets: tensors or arrays of any shape containing integer
            class labels in [0, num_classes). Rows are ground-truth classes,
            columns are predicted classes (row i, col j = number of voxels
            of true class i predicted as class j).
        num_classes: number of facies classes.

    Returns:
        np.ndarray of shape (num_classes, num_classes), dtype int64.
    """
    preds = torch.as_tensor(preds).reshape(-1).long()
    targets = torch.as_tensor(targets).reshape(-1).long()

    indices = targets * num_classes + preds
    cm = torch.bincount(indices, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes).cpu().numpy()


def metrics_from_confusion_matrix(cm, smooth=1e-6):
    """
    Derives per-class precision, recall, F1 and IoU directly from a confusion
    matrix, avoiding a second pass over the (potentially huge) 3D volume.

    Args:
        cm: (num_classes, num_classes) confusion matrix (rows=true, cols=pred).
        smooth: numerical stabilizer to avoid division by zero.

    Returns:
        dict with 'precision', 'recall', 'f1', 'iou' as per-class numpy arrays.
    """
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    pred_totals = cm.sum(axis=0)  # column sums
    true_totals = cm.sum(axis=1)  # row sums
    union = true_totals + pred_totals - tp

    precision = (tp + smooth) / (pred_totals + smooth)
    recall = (tp + smooth) / (true_totals + smooth)
    f1 = (2 * tp + smooth) / (true_totals + pred_totals + smooth)
    iou = (tp + smooth) / (union + smooth)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou,
    }


def binary_metrics_at_threshold(prob_map, target, threshold, smooth=1e-6):
    """
    Computes precision, recall, F1 and IoU for the positive ("Falla") class
    of a binary segmentation problem, given a continuous probability map and
    an arbitrary decision threshold.

    This underpins the threshold sweep requested to test the robustness of
    the model beyond the default 0.5 cut used during validation (see
    mejoras.txt): sweeping `threshold` and re-calling this function traces
    out the full precision/recall trade-off curve.

    Args:
        prob_map: array-like, probability of the positive class, any shape.
        target: array-like of the same shape with binary ground truth (0/1).
        threshold: float in [0, 1], voxels with prob_map >= threshold are
            classified as positive (Falla).
        smooth: numerical stabilizer.

    Returns:
        dict with scalar 'precision', 'recall', 'f1', 'iou', plus the raw
        'tp', 'fp', 'fn', 'tn' voxel counts.
    """
    prob_map = np.asarray(prob_map)
    target = np.asarray(target).astype(bool)

    pred_pos = prob_map >= threshold

    tp = np.logical_and(pred_pos, target).sum(dtype=np.float64)
    fp = np.logical_and(pred_pos, ~target).sum(dtype=np.float64)
    fn = np.logical_and(~pred_pos, target).sum(dtype=np.float64)
    tn = np.logical_and(~pred_pos, ~target).sum(dtype=np.float64)

    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    f1 = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)

    return {
        'precision': float(precision), 'recall': float(recall),
        'f1': float(f1), 'iou': float(iou),
        'tp': float(tp), 'fp': float(fp), 'fn': float(fn), 'tn': float(tn),
    }


def aggregate_metrics(all_metrics):
    """
    Aggregates metrics across batches and computes per-class and global averages.
    """
    # Average across all batches
    avg_metrics = {
        'per_class_precision': np.mean(all_metrics['precision'], axis=0),
        'per_class_recall': np.mean(all_metrics['recall'], axis=0),
        'per_class_f1': np.mean(all_metrics['f1'], axis=0),
        'per_class_iou': np.mean(all_metrics['iou'], axis=0),
    }
    
    # Compute global mean averages across all classes
    avg_metrics['mIoU'] = np.mean(avg_metrics['per_class_iou'])
    avg_metrics['mean_f1'] = np.mean(avg_metrics['per_class_f1'])
    avg_metrics['mean_precision'] = np.mean(avg_metrics['per_class_precision'])
    avg_metrics['mean_recall'] = np.mean(avg_metrics['per_class_recall'])
    
    return avg_metrics

if __name__ == "__main__":
    # Test metrics calculation
    num_classes = 6
    dummy_pred = torch.randint(0, 6, (1, 128, 128, 128))
    dummy_target = torch.randint(0, 6, (1, 128, 128, 128))
    
    res = calculate_metrics(dummy_pred, dummy_target, num_classes=num_classes)
    agg = aggregate_metrics({'precision': [res['precision']], 
                             'recall': [res['recall']], 
                             'f1': [res['f1']], 
                             'iou': [res['iou']]})
    
    print(f"Mean IoU: {agg['mIoU']:.4f}")
    print(f"Global F1 Score: {agg['mean_f1']:.4f}")
