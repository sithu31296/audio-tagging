import torch
import random
import torchaudio
from pathlib import Path
from torch import Tensor
from torchaudio import transforms as T
from torchaudio import functional as AF
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from .transforms import mixup_augment


class UrbanSound8k(Dataset):
    CLASSES = ['air conditioner', 'car horn', 'children playing', 'dog bark', 'drilling', 'engine idling', 'gun shot', 'jackhammer', 'siren', 'street music']

    def __init__(self, split, data_cfg, mixup_cfg=None, transform=None, spec_transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.num_classes = len(self.CLASSES)
        self.transform = transform
        self.spec_transform = spec_transform
        self.mixup = mixup_cfg['MIXUP'] if mixup_cfg is not None else 0.0
        self.mixup_alpha = mixup_cfg['MIXUP_ALPHA'] if mixup_cfg is not None else 0.0
        self.label_smooth = mixup_cfg['SMOOTHING'] if mixup_cfg is not None else 0.0
        self.sample_rate = data_cfg['SAMPLE_RATE']
        self.num_frames = self.sample_rate * data_cfg['AUDIO_LENGTH']

        val_fold = 10

        self.mel_tf = T.MelSpectrogram(self.sample_rate, data_cfg['WIN_LENGTH'], data_cfg['WIN_LENGTH'], data_cfg['HOP_LENGTH'], data_cfg['FMIN'], data_cfg['FMAX'], n_mels=data_cfg['N_MELS'], norm='slaney')    # using mel_scale='slaney' is better
        self.data, self.targets = self.get_data(data_cfg['ROOT'], split, val_fold)
        print(f"Found {len(self.data)} {split} audios in {data_cfg['ROOT']}.")


    def get_data(self, root: str, split: int, fold: int):
        root = Path(root)
        files = (root / 'audio').rglob('*.wav')
        files = list(filter(lambda x: not str(x.parent).endswith(f"fold{fold}") if split == 'train' else str(x.parent).endswith(f"fold{fold}"), files))
        targets = list(map(lambda x: int(x.stem.split('-', maxsplit=2)[1]), files))
        assert len(files) == len(targets)
        return files, targets


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        audio, sr = torchaudio.load(self.data[index]) 
        if audio.shape[0] != 1: audio = audio[:1]           # reduce to mono 
        audio = AF.resample(audio, sr, self.sample_rate)     # resample the audio
        if audio.shape[1] < self.num_frames: audio = torch.cat([audio, torch.zeros(1, self.num_frames-audio.shape[1])], dim=-1)
        target = torch.tensor(self.targets[index])

        if self.transform: audio = self.transform(audio)

        if random.random() < self.mixup:
            next_index = random.randint(0, len(self.data)-1)
            next_audio, sr = torchaudio.load(self.data[next_index])
            if next_audio.shape[0] != 1: next_audio = next_audio[:1]           # reduce to mono 
            next_audio = AF.resample(next_audio, sr, self.sample_rate)
            if next_audio.shape[1] < self.num_frames: next_audio = torch.cat([next_audio, torch.zeros(1, self.num_frames-next_audio.shape[1])], dim=-1)
            next_target = torch.tensor(self.targets[next_index])
            audio, target = mixup_augment(audio, target, next_audio, next_target, self.mixup_alpha, self.num_classes, self.label_smooth)
        else:
            target = F.one_hot(target, self.num_classes).float()

        audio = self.mel_tf(audio)                          # convert to mel spectrogram  
        audio = 10.0 * audio.clamp_(1e-10).log10()          # convert to log mel spectrogram
        
        if self.spec_transform: audio = self.spec_transform(audio)
        
        return audio, target


if __name__ == '__main__':
    data_cfg = {
        'ROOT': 'C:/Users/sithu/Documents/Datasets/Urbansound8K',
        'SAMPLE_RATE': 32000,
        'AUDIO_LENGTH': 4,
        'WIN_LENGTH': 1024,
        'HOP_LENGTH': 320,
        'N_MELS': 64,
        'FMIN': 50,
        'FMAX': 14000
    }
    aug_cfg = {
        'MIXUP': 0.5,
        'MIXUP_ALPHA': 10,
        'SMOOTHING': 0.1
    }
    dataset = UrbanSound8k('train', data_cfg, aug_cfg)
    dataloader = DataLoader(dataset, 2, True)
    for audio, target in dataloader:
        print(audio.shape, target.argmax(dim=1))
        print(audio.min(), audio.max())
        break