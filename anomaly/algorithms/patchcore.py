import torch
import numpy as np

from sklearn.random_projection import SparseRandomProjection
from anomaly.algorithms.sampling_methods import kCenterGreedy


def embedding_concat(embedding_1, embedding_2):
    pass

class Patchcore:
    def __init__(self, feature_extractor, coreset_sampling_ratio, device):
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.device = device
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
    
    def fit(self, data_loader):
        embedding_list = []
        for i, data in enumerate(data_loader):
            features = self.feature_extractor(data)
            embedding = embedding_concat(features[0].numpy(), features[1].asnumpy())
            embedding_list.extend(embedding)

        total_embeddings = np.array(embedding_list, dtype=np.float32)
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9)
        self.randomprojector.fit(total_embeddings)

        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(model=self.randomprojector,
                                             already_selected=[],
                                             N=int(total_embeddings.shape[0] * self.coreset_sampling_ratio))
        self.embedding_coreset = total_embeddings[selected_idx]

        print('initial embedding size : {}'.format(total_embeddings.shape))
        print('final embedding size : {}'.format(self.mbedding_coreset.shape))
    
    def predict(self, data):
        assert len(data.shape) == 4
        assert data.shape[0] == 1
        features = self.feature_extractor(data)
        embedding = embedding_concat(features[0], features[1])
        distances = torch.cdist(embedding, self.embedding_coreset, p=2)
        score_patches, _ = torch.topk(distances, k=9)
        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
        score = w * max(score_patches[:, 0])
        return score
        