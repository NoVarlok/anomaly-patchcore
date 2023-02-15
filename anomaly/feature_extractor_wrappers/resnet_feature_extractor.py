import torch
from anomaly.feature_extractor_wrappers import BaseFeatureExtractor


class ResNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model, device, layers=[2, 3]):
        super().__init__(model, device, layers)
    
    def forward(self, x):
        features = []
        layers = [self.model.layer1, self.model.layer2,
                  self.model.layer3, self.model.layer4]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, layer in enumerate(layers):
            x = layer(x)
            if i + 1 in self.layers:
                features.append(x)
        
        return features
