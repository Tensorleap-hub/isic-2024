import numpy as np
from code_loader.contract.enums import LeapDataType
from code_loader.contract.visualizer_classes import LeapImage, LeapHorizontalBar
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_visualizer

from isic.config import CONFIG
from isic.encoders import MEAN, STD

LABELS = CONFIG['model']['labels']


@tensorleap_custom_visualizer('lesion_tile', LeapDataType.Image)
def lesion_tile(image: np.ndarray) -> LeapImage:
    hwc = image[0].transpose(1, 2, 0) * STD + MEAN
    return LeapImage((np.clip(hwc, 0, 1) * 255).astype(np.uint8))


@tensorleap_custom_visualizer('class_scores', LeapDataType.HorizontalBar)
def class_scores(probs: np.ndarray) -> LeapHorizontalBar:
    return LeapHorizontalBar(body=probs[0].astype(np.float32), labels=LABELS)
