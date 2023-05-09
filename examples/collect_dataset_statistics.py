import anomaly
from anomaly.datasets import createDatasetDataloaders
from anomaly.feature_extractor_wrappers import ResNetFeatureExtractor
from anomaly.algorithms import Patchcore
from anomaly.utils import seed_everything

import numpy as np
import pandas as pd
import os
import sklearn
import argparse
import torch

from sklearn.metrics import roc_auc_score
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Videos to images')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--output_report', type=str,)
    parser.add_argument('--target_good_images', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    categories = [category for category in sorted(os.listdir(args.dataset_dir)) if os.path.isdir(os.path.join(args.dataset_dir, category))]

    processed_categories = []
    aurocs = []

    for category in categories:
        seed_everything(42)
        print('CATEGORY:', category)
        train_loader, test_loader = createDatasetDataloaders(args.dataset_dir, category, 16)
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        feature_extractor = ResNetFeatureExtractor(model, device=device)
        patchcore = Patchcore(feature_extractor, 0.1, device)
        
        patchcore.fit(train_loader)
        scores = []
        labels = []
        for i, data in enumerate(tqdm(test_loader)):
            # img, gt, label, idx = data
            img, label, idx = data
            score = patchcore.predict(img)
            labels.append(int(label[0]))
            scores.append(score)
        
        if args.target_good_images:
            labels = [label ^ 1 for label in labels]
        auroc = roc_auc_score(labels, scores)
        print('AUROC:', auroc)
        print()

        processed_categories.append(category)
        aurocs.append(auroc)

        df = pd.DataFrame(data={'class': ['avg'] + processed_categories,
                                'auroc': [np.mean(aurocs)] + aurocs})
        df.to_csv(args.output_report, index=False)



