import os

import onnxruntime as ort
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test

from isic.config import CONFIG
from isic.data_preprocess import preprocess_func_leap
from isic.encoders import encode_image, encode_gt
from isic.metrics import cross_entropy, malignancy_metrics
from isic.visualizers import lesion_tile, class_scores
from isic.metadata import (metadata_acquisition, metadata_lesion, metadata_diagnosis, metadata_reference,
                           metadata_tile_px)

MODEL_CFG = CONFIG['model']
prediction_type = PredictionTypeHandler('malignancy', MODEL_CFG['labels'], channel_dim=-1)


@tensorleap_load_model([prediction_type])
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_CFG['model_path'])
    return ort.InferenceSession(model_path)


@tensorleap_integration_test()
def integration_test(sample_id: str, subset):
    image = encode_image(sample_id, subset)
    gt = encode_gt(sample_id, subset)
    model = load_model()
    probs = model.run(None, {MODEL_CFG['input_name']: image})[MODEL_CFG['output_index']]

    cross_entropy(gt, probs)
    malignancy_metrics(probs, gt)

    lesion_tile(image)
    class_scores(probs)

    metadata_acquisition(sample_id, subset)
    metadata_lesion(sample_id, subset)
    metadata_diagnosis(sample_id, subset)
    metadata_reference(sample_id, subset)
    metadata_tile_px(sample_id, subset)


if __name__ == '__main__':
    subsets = preprocess_func_leap()
    integration_test(subsets[1].sample_ids[0], subsets[1])
