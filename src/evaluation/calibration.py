import numpy as np


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error for binary classification.

    y_true: true labels, 0/1
    y_prob: predicted probability for positive class
    n_bins: number of confidence bins
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    y_pred = (y_prob >= 0.5).astype(int)
    confidence = np.maximum(y_prob, 1.0 - y_prob)
    correctness = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == n_bins - 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)

        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            avg_confidence = np.mean(confidence[in_bin])
            avg_accuracy = np.mean(correctness[in_bin])
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return float(ece)
