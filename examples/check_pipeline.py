import anomaly
from anomaly.datasets import createDatasetDataloaders
from anomaly.feature_extractor_wrappers import ResNetFeatureExtractor
from anomaly.algorithms import Patchcore

import numpy as np
import os
import sklearn

from sklearn.metrics import roc_auc_score
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


if __name__ == '__main__':
    dataset_dir = '/home/lyakhtin/repos/hse/anomaly/mvtec_anomaly_detection'
    category = 'bottle'
    device = 'cuda'

    train_loader, test_loader = createDatasetDataloaders(dataset_dir, category, 16)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    feature_extractor = ResNetFeatureExtractor(model, device=device)
    patchcore = Patchcore(feature_extractor, 0.1, device)
    
    patchcore.fit(train_loader)
    scores = []
    labels = []
    for i, data in enumerate(tqdm(test_loader)):
        img, gt, label, idx = data
        score = patchcore.predict(img)
        labels.append(int(label[0]))
        scores.append(score)
    
    auroc = roc_auc_score(labels, scores)
    print('AUROC:', auroc)

