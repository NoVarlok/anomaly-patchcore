import torch
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class MVTecDataset():
    def __init__(self, root, preprocessing, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
        
        self.phase = phase
        self.preprocessing = preprocessing
        self.img_paths, self.labels, self.types = self.load_dataset()
    
    def load_dataset(self):
        img_paths = []
        labels = []
        types = []

        defect_types = sorted(os.listdir(self.img_path))
        for defect_type in defect_types:
            defect_dir = os.path.join(self.img_path, defect_type)
            imgs = sorted(os.listdir(defect_dir))
            imgs = [x for x in imgs if x.endswith('.png') or x.endswith('.jpg')]
            imgs = [os.path.join(defect_dir, x) for x in imgs]

            img_paths.extend(imgs)
            types.extend([defect_type] * len(imgs))

            if defect_type == 'good':
                labels.extend([0] * len(imgs))
            else:
                labels.extend([1] * len(imgs))

        return img_paths, labels, types
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        # img = Image.open(img_path).convert('RGB')
        # img = self.preprocessing(img)
        img = self.preprocessing(img_path)

        return img, label, idx
    
def createDatasets(dataset_root_path, category, preprocessing):
    # mean = [0.485, 0.456, 0.406]
    # std = [0.229, 0.224, 0.225]

    # preprocessing = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std)
    # ])

    category_dir = os.path.join(dataset_root_path, category)
    train_dataset = MVTecDataset(category_dir, preprocessing, 'train')
    test_dataset = MVTecDataset(category_dir, preprocessing, 'test')

    return train_dataset, test_dataset

def createDatasetDataloaders(dataset_root_path, category, preprocessing, batch_size):
    train_dataset, test_dataset = createDatasets(dataset_root_path, category, preprocessing)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_dataloader, test_dataloader
