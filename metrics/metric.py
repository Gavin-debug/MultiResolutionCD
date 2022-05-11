import numpy as np

def get_metric(pred, gt):
    pred, gt = pred.detach().cpu().numpy(), gt.detach().cpu().numpy()
    pred_inv, gt_inv = np.logical_not(pred), np.logical_not(gt)
    true_pos = float(np.logical_and(pred, gt).sum())
    true_neg = np.logical_and(pred_inv, gt_inv).sum()
    false_pos = np.logical_and(pred, gt_inv).sum()
    false_neg = np.logical_and(pred_inv, gt).sum()

    pre = true_pos / (true_pos + false_pos + 1e-6)
    rec = true_pos / (true_pos + false_neg + 1e-6)
    acc = (true_neg + true_pos) / (true_pos + true_neg + false_neg + false_pos + 1e-6)
    F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    Iou = true_pos / (true_pos + false_pos + false_neg + 1e-6)

    return {'pre': pre, 'rec': rec, 'acc': acc, 'F1': F1, 'Iou': Iou}