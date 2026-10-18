import io

import h5py
import numpy as np
from PIL import Image
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_input_encoder, tensorleap_gt_encoder

from isic.config import CONFIG

IMG = CONFIG['image']
MEAN = np.array(IMG['mean'], dtype=np.float32)
STD = np.array(IMG['std'], dtype=np.float32)
N_CLASSES = len(CONFIG['model']['labels'])


def read_jpeg_bytes(sample_id: str, preprocess: PreprocessResponse) -> bytes:
    with h5py.File(preprocess.data['images_path'], 'r') as f:
        return f[sample_id][()]


def decode_image(raw: bytes) -> Image.Image:
    return Image.open(io.BytesIO(raw)).convert('RGB')


@tensorleap_input_encoder('image', channel_dim=1)
def encode_image(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    img = decode_image(read_jpeg_bytes(sample_id, preprocess)).resize((IMG['size'], IMG['size']), Image.BILINEAR)
    x = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)


@tensorleap_gt_encoder('malignancy')
def encode_gt(sample_id: str, preprocess: PreprocessResponse) -> np.ndarray:
    target = int(preprocess.data['metadata_df'].loc[sample_id, 'target'])
    return np.eye(N_CLASSES, dtype=np.float32)[target]
