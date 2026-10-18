from typing import Dict

import numpy as np
from code_loader.contract.enums import MetricDirection
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_metric, tensorleap_custom_loss

EPS = 1e-7


@tensorleap_custom_loss('cross_entropy')
def cross_entropy(gt: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return -np.sum(gt * np.log(np.clip(probs, EPS, 1.0)), axis=-1).astype(np.float32)


@tensorleap_custom_metric('malignancy', direction={
    'p_malignant': MetricDirection.Upward,
    'is_correct': MetricDirection.Upward,
    'false_negative': MetricDirection.Downward,
    'false_positive': MetricDirection.Downward,
})
def malignancy_metrics(probs: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray]:
    p = probs[:, 1]
    pred = (p >= 0.5).astype(np.int32)
    target = gt.argmax(axis=-1)
    return {
        'p_malignant': p.astype(np.float32),
        'is_correct': (pred == target).astype(np.int32),
        'false_negative': ((pred == 0) & (target == 1)).astype(np.int32),
        'false_positive': ((pred == 1) & (target == 0)).astype(np.int32),
    }
