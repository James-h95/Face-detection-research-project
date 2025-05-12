import numpy as np
import matplotlib.pyplot as plt

def compute_fppw_missrate(scores, y_true, thresholds):
    fppw_list = []
    miss_rate_list = []

    n_neg = np.sum(y_true == 0)
    n_pos = np.sum(y_true == 1)

    for t in thresholds:
        y_pred = (scores >= t).astype(int)

        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        fppw = fp / n_neg if n_neg != 0 else 0
        miss_rate = fn / n_pos if n_pos != 0 else 0

        fppw_list.append(fppw)
        miss_rate_list.append(miss_rate)

    return np.array(fppw_list), np.array(miss_rate_list)

def plot_multiple_fppw_curves(score_label_pairs, thresholds=None, labels=None, title="Miss Rate vs FPPW Curve"):
    """
    draw multiple SVM output Miss Rate vs FPPW curve
    parameters:
        - score_label_pairs: list of (scores, y_true)
        - thresholds: optional, mannually set thresholds（according to scores set range automatically）
        - labels:default "Model 1"、"Model 2"…）
        - title: title of the figure
    """
    plt.figure(figsize=(8, 6))

    #set label
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(score_label_pairs))]

    #set thresholds range
    if thresholds is None:
        all_scores = np.concatenate([s for s, _ in score_label_pairs])
        thresholds = np.linspace(all_scores.max(), all_scores.min(), 200)

    #plot
    for (scores, y_true), label in zip(score_label_pairs, labels):
        fppw, miss_rate = compute_fppw_missrate(scores, y_true, thresholds)
        plt.semilogx(fppw, miss_rate, label=label, linewidth=1)

    plt.xlabel("False Positives Per Window (log scale)")
    plt.ylabel("Miss Rate")
    plt.title(title)
    plt.grid(True, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()
