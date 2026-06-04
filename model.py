import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config
from torchvision.models import resnet18, ResNet18_Weights


class CosineClassifier(nn.Module):
    def __init__(self, feat_dim, n_classes, scale=16.0):
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.randn(n_classes, feat_dim))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, feat):
        f = F.normalize(feat, dim=1)
        w = F.normalize(self.weight, dim=1)
        return self.scale * f @ w.t()        # cosine logits (B, n_classes)

    @torch.no_grad()
    def append_prototypes(self, protos):
        protos = F.normalize(protos, dim=1).to(self.weight.device)
        self.weight = nn.Parameter(torch.cat([self.weight.data, protos], dim=0))


class FSCILModel(nn.Module):
    def __init__(self, cfg: Config, n_base):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.feat_dim = backbone.fc.in_features   # 512
        backbone.fc = nn.Identity()               # use ResNet as a feature extractor
        self.backbone = backbone
        self.head = CosineClassifier(self.feat_dim, n_base, cfg.cosine_scale)

    def encode(self, x):
        return self.backbone(x)                   # (B, 512)

    def forward(self, x):
        return self.head(self.encode(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval() 