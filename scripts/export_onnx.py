import sys
from pathlib import Path

import timm
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isic.config import CONFIG


# ISICModel is the vit_tiny branch of the 3rd-place solution wrapper:
# https://github.com/kyohei-123/kaggle-isic-2024-3rd-place-solution/blob/main/train_20240830205516.py
class ISICModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False)
        in_features = self.model.head.in_features
        self.model.norm = nn.Identity()
        self.model.fc_norm = nn.Identity()
        self.model.head_drop = nn.Identity()
        self.model.head = nn.Identity()
        self.feature_extractor = nn.Sequential(nn.Flatten(), nn.Linear(in_features, 64), nn.ReLU())
        self.dropouts = nn.ModuleList([nn.Dropout(0.5)])
        self.classifier = nn.Linear(64, 1)

    def forward(self, images):
        features = self.feature_extractor(self.model(images))
        features = torch.mean(torch.stack([dropout(features) for dropout in self.dropouts]), dim=0)
        return self.classifier(features)


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
    size = CONFIG['image']['size']
    base = ISICModel(cfg['backbone'])
    state = torch.load(Path(CONFIG['data']['data_dir']) / cfg['checkpoint'], map_location='cpu', weights_only=True)
    base.load_state_dict(state)
    model = TwoClassProbs(base).eval()
    out_path = root / cfg['model_path']
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
