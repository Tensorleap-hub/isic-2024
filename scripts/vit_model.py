import timm
import torch
from torch import nn


# vit_tiny branch of the 3rd-place solution wrapper:
# https://github.com/kyohei-123/kaggle-isic-2024-3rd-place-solution/blob/main/train_20240830205516.py
class ISICModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
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
