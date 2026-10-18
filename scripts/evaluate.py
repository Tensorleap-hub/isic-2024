import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isic.config import CONFIG
from train import TileDataset, partial_auc, pick_device, val_frame, val_transform
from vit_model import ISICModel


def torch_scorer(checkpoint):
    device = pick_device()
    model = ISICModel(CONFIG['model']['backbone']).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
    model.eval()

    def score(batch):
        with torch.no_grad():
            return model(batch.to(device)).squeeze(-1).cpu().numpy()
    return score


def onnx_scorer(path):
    sess = ort.InferenceSession(path)
    name = sess.get_inputs()[0].name

    def score(batch):
        probs = sess.run(None, {name: batch.numpy()})[CONFIG['model']['output_index']][:, 1]
        return np.log(probs) - np.log1p(-probs)
    return score


def predict(score, images_path, ids, targets):
    loader = DataLoader(TileDataset(images_path, ids, targets, val_transform),
                        batch_size=CONFIG['train']['batch_size_val'], num_workers=CONFIG['train']['num_workers'])
    outputs = np.concatenate([score(x) for x, _ in loader])
    y = np.asarray(targets, dtype=np.float32)
    return {'n': int(len(y)), 'positives': int(y.sum()), 'pauc': float(partial_auc(y, outputs)),
            'auc': float(roc_auc_score(y, outputs))}


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--checkpoint')
    group.add_argument('--onnx')
    parser.add_argument('--fold', type=int, default=CONFIG['data']['validation_fold'])
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    data_dir = Path(CONFIG['data']['data_dir'])
    images_path = str(data_dir / CONFIG['data']['images_file'])
    score = torch_scorer(args.checkpoint) if args.checkpoint else onnx_scorer(args.onnx)

    recipe_val = val_frame(pd.read_parquet(data_dir / CONFIG['data']['folds_file']), args.fold)
    subset = pd.read_parquet(data_dir / CONFIG['data']['subset_file'])
    subset_val = subset[(subset.subset == 'validation') & (subset.fold == args.fold)]

    result = {
        'model': args.checkpoint or args.onnx,
        'fold': args.fold,
        'recipe_val': predict(score, images_path, recipe_val.isic_id, recipe_val.target),
        'subset_val': predict(score, images_path, subset_val.isic_id, subset_val.target),
    }
    out = Path(args.out) if args.out else Path(result['model']).with_suffix('.eval.json')
    out.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
