import torch
import torch.nn.functional as F
from anomaly.feature_extractor_wrappers import BaseFeatureExtractor


class ResNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model, device, layers=[2, 3]):
        super().__init__(model, device, layers)
    
    def forward(self, x):
        features = []
        layers = [self.model.layer1, self.model.layer2,
                  self.model.layer3, self.model.layer4]

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        for i, layer in enumerate(layers):
            x = layer(x)
            if i + 1 in self.layers:
                features.append(x)
        
        features = [F.avg_pool2d(x, 3, 1, 1) for x in features]
        return features
