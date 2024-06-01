import numpy as np

def detect_by_threshold(features, threshold=0.5):
    """
    A simple threshold-based detector that flags samples as poisoned if
    their average pixel intensity is below a certain threshold.
    This is a very basic form of anomaly detection, assuming poisoned samples
    have been darkened.
    """
    suspected_indices = []
    for index, img_features in enumerate(features):
        if np.mean(img_features) < threshold:  # Check if below threshold
            suspected_indices.append(index)
    return suspected_indices
