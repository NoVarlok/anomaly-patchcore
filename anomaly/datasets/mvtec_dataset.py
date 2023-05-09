import torch
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class MVTecDataset():
    def __init__(self, root, transform, gt_transform, phase):
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            # self.gt_path = os.path.join(root, 'ground_truth')
        
        self.phase = phase
        self.transform = transform
        self.gt_transform = gt_transform
        # self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()
        self.img_paths, self.labels, self.types = self.load_dataset()
    
    def load_dataset(self):
        img_paths = []
        # gt_paths = []
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
                # gt_paths.extend([0] * len(imgs))
                labels.extend([0] * len(imgs))
            else:
                # gt_dir = os.path.join(self.gt_path, defect_type)
                # gt_imgs = sorted(os.listdir(gt_dir))
                # gt_imgs = [x for x in gt_imgs if x.endswith('.png') or x.endswith('.jpg')]
                # gt_imgs = [os.path.join(gt_dir, x) for x in gt_imgs]
                # gt_paths.extend(gt_imgs)
                # labels.extend([1] * len(gt_imgs))
                labels.extend([1] * len(imgs))
        
        # assert len(img_paths) == len(gt_paths), "Something wrong with test and ground truth pair!"
        # return img_paths, gt_paths, labels, types
        return img_paths, labels, types
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        # img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        img_path, label, img_type = self.img_paths[idx], self.labels[idx], self.types[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # if gt == 0:
        #     gt = np.zeros((1, np.array(img).shape[-2], np.array(img).shape[-2])).tolist()
        # else:
        #     gt = Image.open(gt)
        #     if self.gt_transform:
        #         gt = self.gt_transform(gt)
        
        # return img, gt, label, idx
        return img, label, idx
    
def createDatasets(dataset_root_path, category):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    gt_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    category_dir = os.path.join(dataset_root_path, category)
    train_dataset = MVTecDataset(category_dir, transform, gt_transform, 'train')
    test_dataset = MVTecDataset(category_dir, transform, gt_transform, 'test')

    return train_dataset, test_dataset

def createDatasetDataloaders(dataset_root_path, category, batch_size):
    train_dataset, test_dataset = createDatasets(dataset_root_path, category)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return train_dataloader, test_dataloader
         
