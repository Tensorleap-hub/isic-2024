import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isic.config import CONFIG


def assign_folds(df, n_folds):
    df['fold'] = -1
    for k, (_, val_idx) in enumerate(GroupKFold(n_splits=n_folds).split(df, df.target, groups=df.patient_id)):
        df.loc[val_idx, 'fold'] = k
    return df


def sample_negatives(part, n_neg, seed):
    neg = part[part.target == 0]
    frac = n_neg / len(neg)
    return neg.groupby('attribution').sample(frac=frac, random_state=seed)


def build_subset(df, is_val, n_neg, seed, name):
    part = df[is_val]
    return pd.concat([part[part.target == 1], sample_negatives(part, n_neg, seed)]).assign(subset=name)


def main():
    cfg = CONFIG['data']
    data_dir = Path(cfg['data_dir'])
    df = pd.read_csv(data_dir / cfg['metadata_file'], low_memory=False)
    df = assign_folds(df, cfg['n_folds'])

    oof = pd.read_csv(data_dir / cfg['reference_oof_file'])
    merged = df[['isic_id', 'fold']].merge(oof[['isic_id', 'fold']], on='isic_id', suffixes=('', '_ref'))
    assert (merged.fold == merged.fold_ref).all()
    df = df.merge(oof[['isic_id', 'oof_prediction']].rename(columns={'oof_prediction': 'ref_oof_logit'}),
                  on='isic_id', how='left')

    is_val = df.fold == cfg['validation_fold']
    val = build_subset(df, is_val, cfg['val_negatives'], cfg['seed'], 'validation')
    train = build_subset(df, ~is_val, cfg['train_negatives'], cfg['seed'], 'training')
    out = pd.concat([train, val]).sample(frac=1, random_state=cfg['seed']).reset_index(drop=True)
    out.to_parquet(data_dir / cfg['subset_file'], index=False)


if __name__ == '__main__':
    main()
