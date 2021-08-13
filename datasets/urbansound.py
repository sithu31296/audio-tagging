import torch
import torchaudio
from pathlib import Path
from torch import Tensor
from torchaudio import transforms as T
from torchaudio import functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class UrbanSound8k(Dataset):
    CLASSES = ['air conditioner', 'car horn', 'children playing', 'dog bark', 'drilling', 'engine idling', 'gun shot', 'jackhammer', 'siren', 'street music']

    def __init__(self, root: str, split: str = 'train', sample_rate: int = 32000, win_length: int = 1024, n_mels: int = 64, fmin: int = 0, fmax: int = None, transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.num_classes = len(self.CLASSES)
        self.sample_rate = sample_rate
        self.transform = transform
        val_fold = 10

        self.mel_tf = T.MelSpectrogram(sample_rate, win_length, win_length, sample_rate//100, fmin, fmax, n_mels=n_mels, norm='slaney', mel_scale='slaney')
        self.data, self.targets = self.get_data(root, split, val_fold)

        print(f"Found {len(self.data)} {split} audios in {root}.")

    def get_data(self, root: str, split: int, fold: int):
        root = Path(root)
        files = (root / 'audio').rglob('*.wav')
        files = list(filter(lambda x: not str(x.parent).endswith(f"fold{fold}") if split == 'train' else str(x.parent).endswith(f"fold{fold}"), files))
        targets = list(map(lambda x: x.stem.split('-', maxsplit=2)[1], files))
        assert len(files) == len(targets)
        return files, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        audio, sr = torchaudio.load(self.data[index]) 
        if audio.shape[0] != 1: audio = audio[:1]           # reduce to mono 
        audio = F.resample(audio, sr, self.sample_rate)     # resample the audio
        if audio.shape[1] < 128000: audio = torch.cat([audio, torch.zeros(1, 128000-audio.shape[1])], dim=-1)
        audio = self.mel_tf(audio)                          # convert to mel spectrogram  
        audio = 10.0 * audio.clamp_(1e-10).log10()          # convert to log mel spectrogram
        if self.transform: audio = self.transform(audio)
        target = int(self.targets[index])
        return audio, target


if __name__ == '__main__':
    dataset = UrbanSound8k('C:\\Users\\sithu\\Documents\\Datasets\\UrbanSound8K', 'val')
    dataloader = DataLoader(dataset, 4, True)
    for audio, target in dataloader: 
        print(audio.shape, target)
        print(audio.min(), audio.max())
        break
