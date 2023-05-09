import torch
import torch.nn.functional as F
import numpy as np

from sklearn.random_projection import SparseRandomProjection
from anomaly.algorithms.sampling_methods import kCenterGreedy


def embedding_concat(embedding_1, embedding_2):
    if embedding_2.shape[-1] > embedding_1.shape[-1]:
        embedding_1, embedding_2 = embedding_2, embedding_1
    
    embedding_1 = F.interpolate(embedding_1, embedding_2.shape[2:], mode='bilinear')
    embedding = torch.cat((embedding_1, embedding_2), dim=1)
    return embedding


def reshape_embedding(embedding):
    embedding_list = []
    for k in range(embedding.shape[0]):
        for i in range(embedding.shape[2]):
            for j in range(embedding.shape[3]):
                embedding_list.append(embedding[k, :, i, j])
    return embedding_list


class Patchcore:
    def __init__(self, feature_extractor, coreset_sampling_ratio, device):
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.device = device
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
    
    @torch.no_grad()
    def fit(self, data_loader):
        embedding_list = []
        
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                # img, gt, label, idx = data
                img, label, idx = data
                img = img.to(self.device)
                features = self.feature_extractor(img)
                embedding = embedding_concat(features[0].cpu(), features[1].cpu()).numpy()
                embedding = reshape_embedding(embedding)
                embedding_list.extend(embedding)

        total_embeddings = np.array(embedding_list, dtype=np.float32)
        self.randomprojector = SparseRandomProjection(n_components='auto', eps=0.9)
        self.randomprojector.fit(total_embeddings)

        selector = kCenterGreedy(total_embeddings, 0, 0)
        selected_idx = selector.select_batch(model=self.randomprojector,
                                             already_selected=[],
                                             N=int(total_embeddings.shape[0] * self.coreset_sampling_ratio))
        self.embedding_coreset = torch.Tensor(total_embeddings[selected_idx])


    @torch.no_grad()
    def predict(self, data):
        assert len(data.shape) == 4
        assert data.shape[0] == 1
        data = data.to(self.device)
        features = self.feature_extractor(data)
        embedding = embedding_concat(features[0], features[1])
        embedding = reshape_embedding(embedding)
        embedding = torch.stack(embedding).cpu()
        distances = torch.cdist(embedding, self.embedding_coreset, p=2)
        score_patches, _ = torch.topk(distances, k=9, largest=False)
        score_patches = score_patches.numpy()
        N_b = score_patches[np.argmax(score_patches[:, 0])]
        w = (1 - (np.max(np.exp(N_b)) / np.sum(np.exp(N_b))))
        score = w * max(score_patches[:, 0])
        return score
        