from pathlib import Path
from .resnet import ResNet

__all__ = {
    "resnet": ResNet,
}

def get_model(model_name: str, model_variant: str, pretrained: str = None, num_classes: int = 50):
    assert model_name in __all__.keys(), f"Unavailable model name >> {model_name}.\nList of available model names: {list(__all__.keys())}"
    if pretrained is not None:
        assert Path(pretrained).exists(), "Please set the correct pretrained model path"
    return __all__[model_name](model_variant, pretrained, num_classes)    