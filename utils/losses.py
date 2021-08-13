import torch
from torch import nn, Tensor
from torch.nn import CrossEntropyLoss


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


__all__ = {
    'ce': CrossEntropyLoss,
    'label_smooth': LabelSmoothCrossEntropy
}


def get_loss(loss_fn_name: str):
    assert loss_fn_name in __all__.keys(), f"Unavailable loss function name >> {loss_fn_name}.\nList of available loss functions: {list(__all__.keys())}"
    return __all__[loss_fn_name]()


if __name__ == '__main__':
    from torch.nn import functional as F
    B = 2
    C = 50
    x = torch.rand(B, C)
    y = torch.randint(0, C-1, (B,))
    loss_fn = CrossEntropyLoss()
    loss_fn2 = LabelSmoothCrossEntropy()
    loss_fn3 = nn.NLLLoss()
    loss = loss_fn(x, y)
    loss2 = loss_fn2(x, y)
    loss3 = loss_fn3(torch.log_softmax(x, dim=-1), y)
    print(loss, loss2, loss3)
