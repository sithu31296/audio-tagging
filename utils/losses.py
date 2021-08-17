import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss, BCELoss, BCEWithLogitsLoss


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = CrossEntropyLoss()

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        target = target.argmax(dim=1)
        loss = self.ce(pred, target)
        return loss


class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        pred = self.log_softmax(pred)
        target = target.argmax(dim=1)
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
    # for audio classification, the following can be used
    'ce': CrossEntropy,
    'label_smooth': LabelSmoothCrossEntropy,

    # for audio tagging and sed, the following can be used
    'bce': BCELoss,
    'bcelogits': BCEWithLogitsLoss,
    'soft_target': SoftTargetCrossEntropy
}


def get_loss(loss_fn_name: str):
    assert loss_fn_name in __all__.keys(), f"Unavailable loss function name >> {loss_fn_name}.\nList of available loss functions: {list(__all__.keys())}"
    return __all__[loss_fn_name]()


if __name__ == '__main__':
    import torch
    from torch.nn import functional as F
    torch.manual_seed(123)
    B = 2
    C = 10
    x = torch.rand(B, C)
    y = torch.rand(B, C)
    loss_fn = CrossEntropy()
    loss = loss_fn(x, y)
    print(loss)
