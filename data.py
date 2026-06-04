import os
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image


IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg")
SPLITS = ("train", "validation")

class NeuDataset(Dataset):
    def __init__(self, split, transforms=None):
        self.transforms = transforms
        self.split = split
        self.classes = split["images"].keys()
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = []
        for cls, files in split["images"].items():
            for fname, img_path in files.items():
                xml_name = fname.replace(".jpg", ".xml")
                xml_path = split["annotations"][xml_name]
                self.samples.append((img_path, xml_path))
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, xml_path = self.samples[idx]
        
        image = Image.open(img_path).convert("RGB")
        if self.transforms:
            image = self.transforms(image)
        
        boxes, labels = self.parse_xml(xml_path)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long)
        }

        return image, target
    
    def parse_xml(self, xml_path):
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
            labels.append(self.class_to_idx[name])
        return boxes, labels


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
                    elif f == "train":
                        for c in os.listdir(os.path.join(root, s, f)):
                            split[s][f][c] = {}
                            for classes in os.listdir(os.path.join(root, s, f, c)):
                                split[s][f][c][classes] = os.path.join(root, s, f, c, classes)
                    else:
                        print("Root Kontrol Et")
        else:
            print("Root Kontrol Et")

    return split

def collate_fn(batch):
    images, targets = zip(*batch)
    return images, targets