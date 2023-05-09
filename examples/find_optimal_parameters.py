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
import tarfile, io

from sklearn.metrics import roc_auc_score
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm
from collections import defaultdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Videos to images')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--grid_search_output_tar', type=str,)
    parser.add_argument('--output_report', type=str,)
    parser.add_argument('--target_good_images', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    categories = [category for category in sorted(os.listdir(args.dataset_dir)) if os.path.isdir(os.path.join(args.dataset_dir, category))]

    results = []
    # Train params grid search
    for use_pca in [False, True]:
        processed_categories = defaultdict(list)
        aurocs = defaultdict(list)
        for category in categories:
            print('CATEGORY:', category)
            seed_everything(42)
            train_loader, test_loader = createDatasetDataloaders(args.dataset_dir, category, 16)
            model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            feature_extractor = ResNetFeatureExtractor(model, device=device)
            train_config = {
                "coreset_sampling_ratio": 0.1,
                "PCA": use_pca,
            }
            patchcore = Patchcore(feature_extractor, train_config, device)
            
            patchcore.fit(train_loader)

            # Inference params grid search
            for p_distance in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                for n_patches in [1, 3, 5, 10, 100]:
                    patchcore.p_distance = p_distance
                    patchcore.n_patches = n_patches
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

                    processed_categories[(p_distance, n_patches)].append(category)
                    aurocs[(p_distance, n_patches)].append(auroc)

        for params in processed_categories:
            p_distance, n_patches = params
            df = pd.DataFrame(data={'class': ['avg'] + processed_categories[params],
                                    'auroc': [np.mean(aurocs[params])] + aurocs[params]})
            df = df.set_index('class')

            results.append({
                "avg_auroc": np.mean(params),
                "use_pca": use_pca,
                "p_distance": p_distance,
                "n_patches": n_patches,
                "dataframe": df,
            })

            print("use_pca:", use_pca)
            print("p_distance:", p_distance)
            print("n_patches:", n_patches)
            print(df.T.to_markdown())
            print()

    with open(args.output_report, "w") as fout:
        results.sort(reverse=False, key=lambda x: x['avg_auroc'])
        best_result = results[0]
        print("use_pca:", best_result["use_pca"], file=fout)
        print("p_distance:", best_result["p_distance"], file=fout)
        print("n_patches:", best_result["n_patches"], file=fout)
        print(best_result["dataframe"].T.to_markdown(), file=fout)
    
    with tarfile.open(args.grid_search_output_tar, "w:gz") as tar:
        data = '\n\n'.join(
            [f"""use_pca: {result['use_pca']}
p_distance: {result['p_distance']}
n_patches: {result['n_patches']}
{result['dataframe'].T.to_markdown()}""" for result in results]
        )
        textIO = io.TextIOWrapper(io.BytesIO(), encoding='utf8')
        textIO.write(data)
        bytesIO = textIO.detach()
        tarinfo = tarfile.TarInfo('results.txt')
        tarinfo.size = bytesIO.tell()
        bytesIO.seek(0)
        tar.addfile(tarinfo, bytesIO)


    

