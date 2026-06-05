import torch
from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    data_root = "./data/CIFAR-100"
    img_size = 32

    base_classes: List[str] = field(default_factory=lambda: [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree',
    ])
    novel_classes: List[str] = field(default_factory=lambda: [
        'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
        'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk',
        'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
        'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television',
        'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe',
        'whale', 'willow_tree', 'wolf', 'woman', 'worm',
    ])
    n_test_per_class = 100
    k_shot = 5

    cosine_scale = 16.0
    base_epoch = 60
    base_lr = 0.01
    batch = 64
    weight_decay = 5e-4
    workers = 4
    
    seed = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"
