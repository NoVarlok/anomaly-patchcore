from abc import ABC, abstractmethod
import torch.nn as nn


class BaseFeatureExtractor(ABC, nn.Module):
    def __init__(self, model, device, layers) -> None:
        super().__init__()
        self.model = model.to(device)
        self.device = device
        self.layers = layers
        self.model.eval()
    
    @abstractmethod
    def forward(self, x):
        ...

