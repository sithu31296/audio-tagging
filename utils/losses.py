from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, BCELoss


class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = self.log_softmax(pred)
        nll_loss = -pred.gather(dim=-1, index=target.unsqueeze(1)).squeeze(1)
        smooth_loss = -pred.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SoftTargetCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = self.log_softmax(pred)
        loss = (-target * pred).sum(dim=-1)
        return loss.mean()


__all__ = {
    'ce': CrossEntropyLoss,
    'bce': BCELoss,
    'label_smooth': LabelSmoothCrossEntropy,
    'soft_target': SoftTargetCrossEntropy
}


def get_loss(loss_fn_name: str):
    assert loss_fn_name in __all__.keys(), f"Unavailable loss function name >> {loss_fn_name}.\nList of available loss functions: {list(__all__.keys())}"
    return __all__[loss_fn_name]()
