import os
import random
import xml.etree.ElementTree as ET

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg")
SPLITS = ("train", "validation")


def scan_dataset(root):
    split = {}
    entries = sorted(os.listdir(root))
    for s in entries:
        if s in SPLITS:
            split[s] = {}
            if os.path.isdir(os.path.join(root, s)):
                for f in os.listdir(os.path.join(root, s)):
                    split[s][f] = {}
                    if f == "annotations":
                        for xml in os.listdir(os.path.join(root, s, f)):
                            split[s][f][xml] = os.path.join(root, s, f, xml)
                    else:
                        for c in os.listdir(os.path.join(root, s, f)):
                            split[s][f][c] = {}
                            for fname in os.listdir(os.path.join(root, s, f, c)):
                                split[s][f][c][fname] = os.path.join(root, s, f, c, fname)
        else:
            print("hatalı ayrım")
    return split


def build_label_map(cfg):
    order = cfg.base_classes + cfg.novel_classes
    return {name: i for i, name in enumerate(order)}


def build_splits(cfg):
    rng = random.Random(cfg.seed)
    raw = scan_dataset(cfg.data_root)
    all_files = {}
    for split_name in SPLITS:
        for cls, files in raw[split_name]["images"].items():
            if cls not in all_files:
                all_files[cls] = []
            all_files[cls].extend(files.values())

    test, train_pool = {}, {}
    for c, paths in all_files.items():
        paths = sorted(paths)
        rng.shuffle(paths)
        test[c] = paths[:cfg.n_test_per_class]
        train_pool[c] = paths[cfg.n_test_per_class:]
    return train_pool, test


def session_samples(train_pool, cfg, rng):
    base = [(p, c) for c in cfg.base_classes for p in train_pool[c]]
    inc = []
    for c in cfg.novel_classes:
        files = train_pool[c][:]
        rng.shuffle(files)
        inc.append([(p, c) for p in files[:cfg.k_shot]])
    return base, inc


def test_samples(test, class_names):
    return [(p, c) for c in class_names for p in test[c]]


def get_transforms(cfg):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    train_t = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(cfg.img_size, padding=8),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_t = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_t, eval_t


class NeuDataset(Dataset):
    def __init__(self, samples, label_map, transforms=None):
        self.samples = samples  # list of (img_path, class_name)
        self.label_map = label_map
        self.transforms = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, cname = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        return image, self.label_map[cname]


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bb = obj.find("bndbox")
        boxes.append([
            int(bb.find("xmin").text),
            int(bb.find("ymin").text),
            int(bb.find("xmax").text),
            int(bb.find("ymax").text),
        ])
        labels.append(name)
    return boxes, labels