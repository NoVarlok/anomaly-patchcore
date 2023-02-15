from abc import ABC, abstractmethod


class BaseFeatureExtractor(ABC):
    def __init__(self, model, device, layers) -> None:
        self.model = model
        self.device = device
        self.layers = layers
    
    @abstractmethod
    def forward(self, x):
        ...

