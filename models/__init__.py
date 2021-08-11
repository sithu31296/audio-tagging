from pathlib import Path
from .cnn14 import CNN14, CNN14DecisionLevelMax

__all__ = {
    "cnn14": CNN14,
    "cnn14decision": CNN14DecisionLevelMax
}


def get_model(model_name: str, num_classes: int = 527):
    assert model_name in __all__.keys(), f"Unavailable model name >> {model_name}.\nList of available model names: {list(__all__.keys())}"
    return __all__[model_name](num_classes)    