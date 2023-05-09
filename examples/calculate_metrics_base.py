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

from sklearn.metrics import roc_auc_score, precision_recall_curve
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
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
    precisions = []
    recalls = []
    f1s = []
    thresholds = []
    fps = []
    fns = []


    for category in categories:
        seed_everything(42)
        print('CATEGORY:', category)
        train_loader, test_loader = createDatasetDataloaders(args.dataset_dir, category, 16)
        model = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
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
            scores = [-score for score in scores]
        auroc = roc_auc_score(labels, scores)
        precision, recall, threshold = precision_recall_curve(labels, scores)
        f1 = 2 * precision * recall / (precision + recall)
        idx_opt = np.argmax(f1)
        precision_opt = precision[idx_opt]
        recall_opt = recall[idx_opt]
        f1_opt = f1[idx_opt]
        threshold_opt = threshold[idx_opt]
        
        fp = 0
        fn = 0
        for label, score in zip(labels, scores):
            if score >= threshold_opt and label == 0:
                fp += 1
            elif score < threshold_opt and label == 1:
                fn += 1
        
        print('AUROC:', auroc)
        print('Precision:', precision_opt)
        print('Recall:', recall_opt)
        print('F1:', f1_opt)
        print('FP:', fp)
        print('FN:', fn)
        print('Threshold:', threshold_opt)

        processed_categories.append(category)
        aurocs.append(auroc)
        precisions.append(precision_opt)
        recalls.append(recall_opt)
        f1s.append(f1_opt)
        fps.append(fp)
        fns.append(fn)
        thresholds.append(threshold_opt)

        df_dict = {
            'class': ['avg'] + processed_categories
        }
        for name, values in zip(['auroc', 'precision', 'recall', 'f1', 'fp', 'fn', 'threshold'],
        [aurocs, precisions, recalls, f1s, fps, fns, thresholds]):
            df_dict[name] = [np.mean(values)] + values

        df = pd.DataFrame(data=df_dict)
        df.to_csv(args.output_report, index=False)
