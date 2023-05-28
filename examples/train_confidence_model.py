import anomaly
from anomaly.utils import seed_everything
from anomaly.datasets import createFineTuningDatasetDataloaders, preprocessings

import numpy as np
import os
import argparse
import torch
import tarfile, io
import yaml

from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Videos to images')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--preprocessing', type=str, default='default', choices=list(preprocessings.PREPROCESSINGS))
    parser.add_argument('--output_model_path', type=str, required=True)
    parser.add_argument('--class_to_id_path', type=str)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset, test_dataset, class2id = createFineTuningDatasetDataloaders(args.dataset_dir,
                                                                               preprocessings.PREPROCESSINGS[args.preprocessing](preprocessings.DEFAULT_IMG_SIZE),
                                                                               args.batch_size)

    model = model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class2id))
    model = model.to(device)
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    
    model.train()
    

