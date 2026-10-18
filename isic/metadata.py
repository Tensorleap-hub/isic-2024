from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DatasetMetadataType
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_metadata

from isic.encoders import read_jpeg_bytes, decode_image

Scalar = Optional[Union[str, int, float, bool]]

LESION_FEATURES = [
    'clin_size_long_diam_mm', 'tbp_lv_areaMM2', 'tbp_lv_minorAxisMM', 'tbp_lv_perimeterMM',
    'tbp_lv_nevi_confidence', 'tbp_lv_dnn_lesion_confidence', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
    'tbp_lv_symm_2axis', 'tbp_lv_eccentricity', 'tbp_lv_deltaLBnorm', 'tbp_lv_color_std_mean',
    'tbp_lv_radial_color_std_max', 'tbp_lv_stdL', 'tbp_lv_H', 'tbp_lv_L',
]


def row(sample_id: str, preprocess: PreprocessResponse) -> pd.Series:
    return preprocess.data['metadata_df'].loc[sample_id]


def nullable_str(value) -> Optional[str]:
    return None if pd.isna(value) else str(value)


def nullable_float(value) -> Optional[float]:
    return None if pd.isna(value) else float(value)


ACQUISITION_TYPES = {
    'patient_id': DatasetMetadataType.string,
    'attribution': DatasetMetadataType.string,
    'tile_type': DatasetMetadataType.string,
    'sex': DatasetMetadataType.string,
    'age': DatasetMetadataType.float,
    'anatom_site': DatasetMetadataType.string,
    'location_simple': DatasetMetadataType.string,
    'sex_missing': DatasetMetadataType.boolean,
    'age_missing': DatasetMetadataType.boolean,
    'site_missing': DatasetMetadataType.boolean,
    'fold': DatasetMetadataType.int,
}

DIAGNOSIS_TYPES = {
    'target': DatasetMetadataType.int,
    'iddx_1': DatasetMetadataType.string,
    'iddx_2': DatasetMetadataType.string,
    'iddx_full': DatasetMetadataType.string,
    'has_lesion_id': DatasetMetadataType.boolean,
}


@tensorleap_metadata('acquisition', ACQUISITION_TYPES)
def metadata_acquisition(sample_id: str, preprocess: PreprocessResponse) -> Dict[str, Scalar]:
    r = row(sample_id, preprocess)
    return {
        'patient_id': str(r.patient_id),
        'attribution': str(r.attribution),
        'tile_type': str(r.tbp_tile_type),
        'sex': nullable_str(r.sex),
        'age': nullable_float(r.age_approx),
        'anatom_site': nullable_str(r.anatom_site_general),
        'location_simple': str(r.tbp_lv_location_simple),
        'sex_missing': bool(pd.isna(r.sex)),
        'age_missing': bool(pd.isna(r.age_approx)),
        'site_missing': bool(pd.isna(r.anatom_site_general)),
        'fold': int(r.fold),
    }


@tensorleap_metadata('lesion', {name: DatasetMetadataType.float for name in LESION_FEATURES})
def metadata_lesion(sample_id: str, preprocess: PreprocessResponse) -> Dict[str, Scalar]:
    r = row(sample_id, preprocess)
    return {name: nullable_float(r[name]) for name in LESION_FEATURES}


@tensorleap_metadata('diagnosis', DIAGNOSIS_TYPES)
def metadata_diagnosis(sample_id: str, preprocess: PreprocessResponse) -> Dict[str, Scalar]:
    r = row(sample_id, preprocess)
    return {
        'target': int(r.target),
        'iddx_1': str(r.iddx_1),
        'iddx_2': nullable_str(r.iddx_2),
        'iddx_full': str(r.iddx_full),
        'has_lesion_id': bool(pd.notna(r.lesion_id)),
    }


@tensorleap_metadata('reference')
def metadata_reference(sample_id: str, preprocess: PreprocessResponse) -> Dict[str, Scalar]:
    logit = float(row(sample_id, preprocess).ref_oof_logit)
    return {'ref_oof_logit': logit, 'ref_oof_prob': float(1.0 / (1.0 + np.exp(-logit)))}


@tensorleap_metadata('tile_px')
def metadata_tile_px(sample_id: str, preprocess: PreprocessResponse) -> int:
    return decode_image(read_jpeg_bytes(sample_id, preprocess)).size[0]
