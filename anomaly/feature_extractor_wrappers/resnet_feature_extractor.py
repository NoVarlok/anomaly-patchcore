import torch
import torch.nn.functional as F
import numpy as np

from anomaly.feature_extractor_wrappers import BaseFeatureExtractor


def get_kernel(output_channels, kernel_size=3):
    margin = kernel_size//2
    kernel = []

    for i in range(kernel_size):
        kernel_row = []
        for j in range(kernel_size):
            kernel_elem = 1 - max(abs(margin - i), abs(margin - j)) * 0.2
            kernel_row.append(kernel_elem)
        kernel.append(kernel_row)
    kernel_sq = np.array(kernel)
    kernel = np.zeros((output_channels, output_channels, kernel_size, kernel_size))
    for i in range(output_channels):
        kernel[i][i] = kernel_sq
    return torch.tensor(np.float32(kernel))


class ResNetFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, model, device, layers=[2, 3], aggregation_fn='avg'):
        super().__init__(model, device, layers)
        self.aggregation_fn = aggregation_fn
        self.out_size = [256, 512, 1024, 2048]
    
    def forward(self, x):
        features = []
        kernels = []
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
                kernels.append(get_kernel(self.out_size[i]).to(self.device))
        
        if self.aggregation_fn == 'avg':
            features = [F.avg_pool2d(x, 3, 1, 1) for x in features]
        elif self.aggregation_fn == 'conv':
            features = [F.conv2d(F.pad(x, (0, 0, 1, 1), "constant", 0), weight) for x, weight in zip(features, kernels)]
        return features
