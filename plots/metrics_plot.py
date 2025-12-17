#!/usr/bin/env python3
"""
Compute anomaly detection / binary-classification metrics from ground truth labels and prediction scores.
Saves ROC curve, Precision-Recall curve, and a summary bar chart of key metrics.

Usage:
  python plots/metrics_plot.py --y-true y_true.npy --y-score y_score.npy --outdir plots/output
If the input files are not provided the script will generate a synthetic example dataset and run on it.

This script does not require scikit-learn; all metrics are computed with numpy.
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt


def roc_curve_from_scores(y_true, y_score):
    # returns fpr, tpr, thresholds
    desc_idx = np.argsort(-y_score)
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        # Degenerate case
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([np.inf, -np.inf])

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    tpr = np.concatenate(([0.0], tps / P, [1.0]))
    fpr = np.concatenate(([0.0], fps / N, [1.0]))

    # thresholds: choose unique score boundaries (include +inf and -inf)
    thresholds = np.concatenate(([np.inf], y_score_sorted, [-np.inf]))
    return fpr, tpr, thresholds


def auc_from_curve(x, y):
    # simple trapezoidal rule
    return np.trapz(y, x)


def precision_recall_curve_from_scores(y_true, y_score):
    desc_idx = np.argsort(-y_score)
    y_true_sorted = y_true[desc_idx]

    tps = np.cumsum(y_true_sorted == 1)
    preds = np.arange(1, len(y_true_sorted) + 1)
    precision = tps / preds
    recall = tps / (np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 1)

    precision = np.concatenate(([1.0], precision, [0.0]))
    recall = np.concatenate(([0.0], recall, [1.0]))
    thresholds = np.concatenate(([np.inf], np.sort(y_score)[::-1], [-np.inf]))
    return precision, recall, thresholds


def confusion_counts(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    TP = int(np.sum((y_pred == 1) & (y_true == 1)))
    FP = int(np.sum((y_pred == 1) & (y_true == 0)))
    TN = int(np.sum((y_pred == 0) & (y_true == 0)))
    FN = int(np.sum((y_pred == 0) & (y_true == 1)))
    return TP, FP, TN, FN


def r_precision(y_true, y_score):
    # R-precision: precision in top-R retrieved, where R = number of relevant items
    R = int(np.sum(y_true == 1))
    if R == 0:
        return 0.0
    idx = np.argsort(-y_score)[:R]
    return float(np.sum(y_true[idx] == 1)) / R


def compute_all_metrics(y_true, y_score, threshold=None):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    fpr, tpr, roc_th = roc_curve_from_scores(y_true, y_score)
    auc = auc_from_curve(fpr, tpr)

    precision, recall, pr_th = precision_recall_curve_from_scores(y_true, y_score)
    # Average precision approximated by area under precision-recall curve (interpolated)
    ap = auc_from_curve(recall[::-1], precision[::-1])

    if threshold is None:
        # choose threshold at score that yields highest F1 on the discrete set
        best_f1 = -1
        best_thr = 0.5
        for thr in np.unique(y_score):
            TP, FP, TN, FN = confusion_counts(y_true, y_score, thr)
            prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            rec = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        threshold = float(best_thr)

    TP, FP, TN, FN = confusion_counts(y_true, y_score, threshold)
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0
    precision_at_thr = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_at_thr = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_at_thr = 2 * precision_at_thr * recall_at_thr / (precision_at_thr + recall_at_thr) if (precision_at_thr + recall_at_thr) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    rprec = r_precision(y_true, y_score)

    metrics = {
        'AUC': float(auc),
        'Average_Precision': float(ap),
        'Threshold_used': float(threshold),
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'Accuracy': float(accuracy),
        'Precision': float(precision_at_thr),
        'Recall': float(recall_at_thr),
        'F1': float(f1_at_thr),
        'Specificity': float(specificity),
        'R-Precision': float(rprec)
    }
    return metrics, (fpr, tpr, roc_th), (precision, recall, pr_th)


def save_plots(outdir, metrics, roc_data, pr_data):
    os.makedirs(outdir, exist_ok=True)
    fpr, tpr, _ = roc_data
    precision, recall, _ = pr_data

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'AUC = {metrics["AUC"]:.3f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    roc_path = os.path.join(outdir, 'roc_curve.png')
    plt.savefig(roc_path)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f'AP = {metrics["Average_Precision"]:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    pr_path = os.path.join(outdir, 'pr_curve.png')
    plt.savefig(pr_path)
    plt.close()

    # Summary bar chart for selected scalar metrics
    keys = ['AUC', 'Average_Precision', 'Accuracy', 'Precision', 'Recall', 'F1', 'R-Precision', 'Specificity']
    vals = [metrics.get(k, 0) for k in keys]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(keys, vals, color='tab:blue')
    plt.ylim(0, 1)
    for bar, v in zip(bars, vals):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f'{v:.3f}', ha='center')
    plt.title('Metrics Summary')
    summary_path = os.path.join(outdir, 'metrics_summary.png')
    plt.savefig(summary_path)
    plt.close()

    return roc_path, pr_path, summary_path


def flatten_inputs(y_true_input, y_score_input):
    # If inputs are lists of arrays (per-video), flatten them
    y_true = np.load(y_true_input) if isinstance(y_true_input, str) else np.asarray(y_true_input)
    y_score = np.load(y_score_input) if isinstance(y_score_input, str) else np.asarray(y_score_input)
    # If loaded arrays are object or list-like, attempt to flatten
    if y_true.dtype == object:
        y_true = np.concatenate([np.asarray(x).ravel() for x in y_true])
    if y_score.dtype == object:
        y_score = np.concatenate([np.asarray(x).ravel() for x in y_score])
    return y_true.ravel(), y_score.ravel()


def generate_synthetic(n=1000, anomaly_frac=0.05, seed=42):
    rng = np.random.RandomState(seed)
    y_true = rng.choice([0, 1], size=n, p=[1 - anomaly_frac, anomaly_frac])
    # scores: anomalies tend to have higher scores
    scores = rng.normal(loc=0.2, scale=0.1, size=n)
    scores[y_true == 1] += 0.6
    scores = np.clip(scores, 0, 1)
    return y_true.astype(int), scores.astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--y-true', type=str, default=None, help='Path to numpy .npy file with ground truth (0/1)')
    parser.add_argument('--y-score', type=str, default=None, help='Path to numpy .npy file with predicted scores (higher = more anomalous)')
    parser.add_argument('--outdir', type=str, default='plots/output', help='Output folder for plots and JSON metrics')
    parser.add_argument('--threshold', type=float, default=None, help='Optional threshold for classification')
    args = parser.parse_args()

    if args.y_true and args.y_score:
        y_true, y_score = flatten_inputs(args.y_true, args.y_score)
    else:
        print('No inputs provided — generating synthetic example data')
        y_true, y_score = generate_synthetic()

    metrics, roc_data, pr_data = compute_all_metrics(y_true, y_score, threshold=args.threshold)
    os.makedirs(args.outdir, exist_ok=True)
    metrics_path = os.path.join(args.outdir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    roc_path, pr_path, summary_path = save_plots(args.outdir, metrics, roc_data, pr_data)

    print('Metrics saved to:', metrics_path)
    print('ROC curve saved to:', roc_path)
    print('PR curve saved to:', pr_path)
    print('Summary plot saved to:', summary_path)
    print('Metrics:')
    for k, v in metrics.items():
        print(f'  {k}: {v}')

if __name__ == '__main__':
    main()
