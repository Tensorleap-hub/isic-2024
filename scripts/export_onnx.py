import argparse
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isic.config import CONFIG
from vit_model import ISICModel


class TwoClassProbs(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, images):
        p = torch.sigmoid(self.base(images))
        return torch.cat([1 - p, p], dim=1)


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = CONFIG['model']
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=str(Path(CONFIG['data']['data_dir']) / cfg['checkpoint']))
    parser.add_argument('--out', default=str(root / cfg['model_path']))
    args = parser.parse_args()
    size = CONFIG['image']['size']
    base = ISICModel(cfg['backbone'])
    base.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
    model = TwoClassProbs(base).eval()
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, size, size),
        str(out_path),
        input_names=[cfg['input_name']],
        output_names=['probs'],
        dynamic_axes={cfg['input_name']: {0: 'batch'}, 'probs': {0: 'batch'}},
        opset_version=17,
        dynamo=False,
    )


if __name__ == '__main__':
    main()
