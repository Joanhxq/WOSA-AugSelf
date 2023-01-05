import cv2
import os
import torch
import numpy as np
from torch.utils import data
from torchvision import transforms


def get_one_hot(label, num_classes=2):
    device = label.device
    size = list(label.size())
    label = label.view(-1)
    ones = torch.sparse.torch.eye(num_classes, device=device)
    ones = ones.index_select(0, label)
    size.append(num_classes)
    label = ones.view(*size[1:])
    return label.permute(2, 0, 1)


class FundusDataset(data.Dataset):
    def __init__(self, img_lst, cfg, transform=None):
        self.img_lst = img_lst
        self.cfg = cfg
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_lst[index]
        lab_path = os.path.join(self.cfg.label_dir, os.path.basename(img_path))
        image = cv2.imread(img_path)
        label = cv2.imread(lab_path, 0)
        image = cv2.resize(image, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (self.cfg.image_size, self.cfg.image_size), interpolation=cv2.INTER_NEAREST)
        if self.transform is not None:
            image, label = self.transform(image, label)
        for k, v in self.cfg.class_maps.items():
            label[label==k] = v
        image, label = np.float32(image), np.float32(label)
        image_mean = np.mean(image, axis=(0, 1), keepdims=True)
        image_std = np.std(image, axis=(0, 1), keepdims=True)
        image_norm = (image - image_mean) / image_std
        image_norm, label = transforms.ToTensor()(image_norm), transforms.ToTensor()(label)
        label = get_one_hot(label.long(), num_classes=len(self.cfg.class_names)+1)
        return image_norm, label

    def __len__(self):
        return len(self.img_lst)


class GrayDataset(data.Dataset):
    def __init__(self, img_lst, cfg, transform=None):
        self.img_lst = img_lst
        self.cfg = cfg
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_lst[index]
        lab_path = img_path.replace("image", "label")
        typ = img_path.split('/')[-2].split('_')[0]
        image = cv2.imread(img_path, 0)
        label = cv2.imread(lab_path, 0)
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, self.size, interpolation=cv2.INTER_NEAREST)
        if self.transform is not None:
            image, label = self.transform(image, label)
        for k, v in self.cfg.class_maps.items():
            label[label==k] = v
        image, label = np.float32(image), np.float32(label)
        image_mean = np.mean(image, axis=(0, 1), keepdims=True)
        image_std = np.std(image, axis=(0, 1), keepdims=True)
        image_norm = (image - image_mean) / image_std
        image_norm, label = transforms.ToTensor()(image_norm), transforms.ToTensor()(label)
        image_norm = torch.cat([image_norm, image_norm, image_norm])
        label = get_one_hot(label.long(), num_classes=self.cfg.num_classes)
        return image_norm, label

    def __len__(self):
        return len(self.img_lst)


