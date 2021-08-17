from torch import distributed as dist
from torch.utils.data import DistributedSampler, SequentialSampler, RandomSampler
from .esc50 import ESC50
from .audioset import AudioSet
from .fsdkaggle import FSDKaggle2018
from .urbansound import UrbanSound8k
from .speechcommands import SpeechCommandsv1
from .speechcommands import SpeechCommandsv2


__all__ = {
    "esc50": ESC50,
    "audioset": AudioSet,
    "fsdkaggle2018": FSDKaggle2018,
    "urbansound8k": UrbanSound8k,
    "speechcommandsv1": SpeechCommandsv1,
    "speechcommandsv2": SpeechCommandsv2
}


def get_train_dataset(dataset_cfg, aug_cfg, transform, spec_transform):
    dataset_name = dataset_cfg['NAME']
    assert dataset_name in __all__.keys(), f"Unavailable dataset name >> {dataset_name}.\nList of available datasets: {list(__all__.keys())}"
    return __all__[dataset_name]('train', dataset_cfg, aug_cfg, transform, spec_transform)

def get_val_dataset(dataset_cfg):
    dataset_name = dataset_cfg['NAME']
    assert dataset_name in __all__.keys(), f"Unavailable dataset name >> {dataset_name}.\nList of available datasets: {list(__all__.keys())}"
    return __all__[dataset_name]('val', dataset_cfg)


def get_sampler(ddp_enable, train_dataset, val_dataset):
    if not ddp_enable:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset, dist.get_world_size(), dist.get_rank(), shuffle=True)
    val_sampler = SequentialSampler(val_dataset)
    return train_sampler, val_sampler