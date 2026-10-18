import argparse
import csv
import io
import random
import sys
from datetime import datetime
from pathlib import Path

import albumentations as A
import h5py
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isic.config import CONFIG
from vit_model import ISICModel

IMG = CONFIG['image']
TRAIN = CONFIG['train']
MEAN, STD = IMG['mean'], IMG['std']

train_transform = A.Compose([
    A.Transpose(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, p=0.75),
    A.OneOf([
        A.MotionBlur(blur_limit=5),
        A.MedianBlur(blur_limit=5),
        A.GaussianBlur(blur_limit=5),
        A.GaussNoise(var_limit=(5.0, 30.0)),
    ], p=0.7),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.0),
        A.GridDistortion(num_steps=5, distort_limit=1.0),
        A.ElasticTransform(alpha=3),
    ], p=0.7),
    A.CLAHE(clip_limit=4.0, p=0.7),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
    A.Resize(IMG['size'], IMG['size']),
    A.CoarseDropout(max_height=int(IMG['size'] * 0.375), max_width=int(IMG['size'] * 0.375), min_holes=1, p=0.7),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])

val_transform = A.Compose([
    A.Resize(IMG['size'], IMG['size']),
    A.Normalize(mean=MEAN, std=STD),
    ToTensorV2(),
])


class TileDataset(Dataset):
    def __init__(self, images_path, isic_ids, targets, transform):
        self.images_path = images_path
        self.isic_ids = list(isic_ids)
        self.targets = list(targets)
        self.transform = transform

    def __len__(self):
        return len(self.isic_ids)

    def __getitem__(self, idx):
        with h5py.File(self.images_path, 'r') as f:
            raw = f[self.isic_ids[idx]][()]
        img = np.array(Image.open(io.BytesIO(raw)))
        return self.transform(image=img)['image'], np.float32(self.targets[idx])


def interleave(positives, negatives, ratio):
    positives, negatives = list(positives), list(negatives)
    random.shuffle(positives)
    random.shuffle(negatives)
    order = []
    p = n = 0
    while p < len(positives) or n < len(negatives):
        if p < len(positives):
            order.append(positives[p])
            p += 1
        for _ in range(ratio):
            if n < len(negatives):
                order.append(negatives[n])
                n += 1
    return order


def partial_auc(y_true, y_hat, min_tpr=0.8):
    max_fpr = 1 - min_tpr
    scaled = roc_auc_score(np.abs(y_true - 1), 1.0 - y_hat, max_fpr=max_fpr)
    return 0.5 * max_fpr ** 2 + (max_fpr - 0.5 * max_fpr ** 2) / 0.5 * (scaled - 0.5)


def epoch_train_frame(df, fold, epoch):
    train_df = df[df.fold != fold]
    pos = train_df[train_df.target == 1]
    if TRAIN['negative_pool'] == 'confident':
        pool = train_df[(train_df.target == 0) & (train_df.tbp_lv_dnn_lesion_confidence > TRAIN['val_confidence_min'])]
    else:
        pool = train_df
    neg = pool.sample(len(pos) * TRAIN['ratio_upsampling'] * TRAIN['ratio_int'], random_state=epoch)
    pos = pd.concat([pos] * TRAIN['ratio_upsampling'])
    return pos, neg


def val_frame(df, fold):
    val_df = df[df.fold == fold]
    pos = val_df[val_df.target == 1]
    neg = val_df[(val_df.target == 0) & (val_df.tbp_lv_dnn_lesion_confidence > TRAIN['val_confidence_min'])]
    neg = neg.sample(len(pos) * TRAIN['ratio_int_val'], random_state=42)
    return pd.concat([pos, neg]).reset_index(drop=True)


def optimizer_params(model):
    no_decay = ['bias', 'norm']
    groups = [{
        'params': [p for n, p in model.named_parameters() if 'model' not in n],
        'weight_decay': TRAIN['weight_decay'],
        'lr': TRAIN['lr'],
    }]
    layers = [model.model.patch_embed] + list(model.model.blocks)
    layers.reverse()
    lr = TRAIN['lr'] * TRAIN['lr_decay_rate']
    for layer in layers:
        groups.append({
            'params': [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': TRAIN['weight_decay'],
            'lr': lr,
        })
        groups.append({
            'params': [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': lr,
        })
        lr *= TRAIN['lr_decay_rate']
    return groups


def build_scheduler(optimizer, total_steps):
    warmup_steps = int(total_steps * TRAIN['warmup_ratio'])
    warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: step / warmup_steps)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=TRAIN['min_lr'])
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def pick_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets_all, outputs_all = [], []
    with torch.set_grad_enabled(training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs).squeeze(-1)
            loss = criterion(outputs, targets)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
            total_loss += loss.item()
            targets_all.append(targets.detach().cpu())
            outputs_all.append(outputs.detach().cpu())
    return total_loss / len(loader), torch.cat(targets_all).numpy(), torch.cat(outputs_all).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=CONFIG['data']['validation_fold'])
    parser.add_argument('--epochs', type=int, default=TRAIN['num_epochs'])
    parser.add_argument('--out-dir', default=None)
    args = parser.parse_args()

    random.seed(TRAIN['seed'])
    np.random.seed(TRAIN['seed'])
    torch.manual_seed(TRAIN['seed'])

    data_dir = Path(CONFIG['data']['data_dir'])
    images_path = str(data_dir / CONFIG['data']['images_file'])
    df = pd.read_parquet(data_dir / CONFIG['data']['folds_file'])
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / TRAIN['runs_dir'] / datetime.now().strftime('%Y%m%d%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    model = ISICModel(CONFIG['model']['backbone'], pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(optimizer_params(model), lr=TRAIN['lr'])
    n_folds = CONFIG['data']['n_folds']
    total_steps = int((TRAIN['ratio_int'] + 1) * int(df.target.sum()) * TRAIN['ratio_upsampling'] * args.epochs
                      * ((n_folds - 1) / n_folds) / TRAIN['batch_size'])
    scheduler = build_scheduler(optimizer, total_steps)

    val_df = val_frame(df, args.fold)
    val_loader = DataLoader(TileDataset(images_path, val_df.isic_id, val_df.target, val_transform),
                            batch_size=TRAIN['batch_size_val'], shuffle=False, num_workers=TRAIN['num_workers'])

    best_score = -np.inf
    no_improve = 0
    with open(out_dir / 'log.csv', 'w', newline='') as f:
        log = csv.writer(f)
        log.writerow(['epoch', 'train_loss', 'val_loss', 'val_pauc', 'best_pauc', 'lr'])
        for epoch in range(args.epochs):
            pos, neg = epoch_train_frame(df, args.fold, epoch)
            order = interleave(pos.isic_id, neg.isic_id, TRAIN['ratio_int'])
            targets = pd.concat([pos, neg]).drop_duplicates('isic_id').set_index('isic_id').target.loc[order]
            train_loader = DataLoader(TileDataset(images_path, order, targets, train_transform),
                                      batch_size=TRAIN['batch_size'], shuffle=False, num_workers=TRAIN['num_workers'])
            train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer, scheduler)
            val_loss, val_targets, val_outputs = run_epoch(model, val_loader, criterion, device)
            score = partial_auc(val_targets, val_outputs)
            if score > best_score:
                best_score = score
                no_improve = 0
                torch.save(model.state_dict(), out_dir / f'fold_{args.fold}_epoch_{epoch + 1}_score_{score:.4f}.pth')
                np.save(out_dir / 'val_outputs.npy', np.stack([val_targets, val_outputs]))
            else:
                no_improve += 1
            log.writerow([epoch + 1, train_loss, val_loss, score, best_score, optimizer.param_groups[0]['lr']])
            f.flush()
            if no_improve > TRAIN['early_stop_count']:
                break


if __name__ == '__main__':
    main()
