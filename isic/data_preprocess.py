from pathlib import Path
from typing import List

import pandas as pd
from code_loader.contract.datasetclasses import PreprocessResponse
from code_loader.contract.enums import DataStateType
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_preprocess

from isic.config import CONFIG

STATES = {'training': DataStateType.training, 'validation': DataStateType.validation}


@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    cfg = CONFIG['data']
    data_dir = Path(cfg['data_dir'])
    df = pd.read_parquet(data_dir / cfg['subset_file'])
    images_path = str(data_dir / cfg['images_file'])
    responses = []
    for name, state in STATES.items():
        part = df[df.subset == name].set_index('isic_id', drop=False)
        responses.append(PreprocessResponse(
            sample_ids=part.index.tolist(),
            data={'metadata_df': part, 'images_path': images_path},
            state=state,
        ))
    return responses
