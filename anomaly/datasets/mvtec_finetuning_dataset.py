import torch
import os
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image


class MVTecFineTuningDataset():
    """MVTecDataset"""
    def __init__(self, mvtec_dir, image_preprocessing, train=False):
        self.image_preprocessing = image_preprocessing
        self.img_paths, self.labels, self.class2id = self.load_dataset(mvtec_dir, train)
    
    def load_dataset(self, mvtec_dir, train):
        """load_dataset"""
        classes = sorted(os.listdir(mvtec_dir))
        classes = [class_name for class_name in classes if os.path.isdir(os.path.join(mvtec_dir, class_name))]
        class2id = {class_name:i for i, class_name in enumerate(classes)}
        image_paths = []
        labels = []
        dataset_type = 'train' if train else 'test'
        for i, class_name in enumerate(classes):
            class_path = os.path.join(mvtec_dir, class_name, dataset_type, 'good')
            images = sorted(os.listdir(class_path))
            images = [image for image in images if image.endswith('.png') or image.endswith('.jpg')]
            for image in images:
                labels.append(i)
                image_paths.append(os.path.join(class_path, image))
        return image_paths, labels, class2id

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = self.image_preprocessing(img_path)
        label = self.labels[idx]
        return img, label


def createFineTuningDatasets(dataset_root_path, preprocessing):
    train_dataset = MVTecFineTuningDataset(dataset_root_path, preprocessing, True)
    test_dataset = MVTecFineTuningDataset(dataset_root_path, preprocessing, False)

    return train_dataset, test_dataset

def createFineTuningDatasetDataloaders(dataset_root_path, preprocessing, batch_size):
    train_dataset, test_dataset = createFineTuningDatasets(dataset_root_path, preprocessing)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    assert train_dataset.class2id == test_dataset.class2id
    return train_dataloader, test_dataloader, train_dataset.class2id
