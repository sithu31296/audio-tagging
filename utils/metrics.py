import torch
import math
from torch import Tensor
from sklearn import metrics as skmetrics
from scipy import stats


class Metrics:
    def __init__(self, metric='accuracy') -> None:
        assert metric in ['accuracy', 'mAP', 'map']
        self.metric = metric
        self.count = 0
        self.acc = 0.0
        self.preds = []
        self.targets = []

    def _update_accuracy(self, pred: Tensor, target: Tensor):
        self.acc += (pred.argmax(dim=-1) == target.argmax(dim=-1)).sum(dim=0).item()
        self.count += target.shape[0]

    def _update_mAP(self, pred: Tensor, target: Tensor):
        self.preds.append(pred)
        self.targets.append(target)

    def _compute_accuracy(self):
        return [(self.acc / self.count) * 100]

    def _compute_mAP(self):
        preds = torch.cat(self.preds, dim=0).cpu().numpy()
        targets = torch.cat(self.targets, dim=0).cpu().numpy()
        ap = skmetrics.average_precision_score(targets, preds, average=None)
        auc = skmetrics.roc_auc_score(targets, preds, average=None)
        mAP = ap.mean()
        mAUC = auc.mean()
        d_prime = stats.norm().ppf(mAUC) * math.sqrt(2.0)
        return mAP * 100, mAUC * 100, d_prime * 100
    
    def update(self, pred: Tensor, target: Tensor):
        if self.metric == 'accuracy':
            self._update_accuracy(pred, target)
        else:
            self._update_mAP(pred, target)

    def compute(self):
        if self.metric == 'accuracy':
            return self._compute_accuracy()
        else:
            return self._compute_mAP()