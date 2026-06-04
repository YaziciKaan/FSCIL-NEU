import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    data_root = "./data/NEU-DET"
    img_size = 200

    base_classes: List[str] = field(default_factory=lambda: ["RS", "Pa", "Cr", "PS"])
    novel_classes: List[str] = field(default_factory=lambda: ["In", "Sc"])
    n_test_per_class = 60
    k_shot = 5

    cosine_scale = 16.0
    base_epoch = 60
    base_lr = 0.01
    batch = 64
    weight_decay = 5e-4
    workers = 4
    
    seed = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"
