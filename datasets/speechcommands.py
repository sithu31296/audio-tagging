import torch
import torchaudio
import os
from pathlib import Path
from torch import Tensor
from torchaudio import transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class SpeechCommandsv1(Dataset):
    CLASSES = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']

    def __init__(self, root: str, split: str = 'train', sample_rate: int = 32000, win_length: int = 1024, n_mels: int = 64, fmin: int = 0, fmax: int = None, transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.num_classes = len(self.CLASSES)
        self.transform = transform

        self.mel_tf = T.MelSpectrogram(sample_rate, win_length, win_length, sample_rate//100, fmin, fmax, n_mels=n_mels, norm='slaney', mel_scale='slaney')
        self.resample = T.Resample(16000, sample_rate)
        self.data, self.targets = self.get_data(root, split)

        print(f"Found {len(self.data)} {split} audios in {root}.")

    def get_data(self, root: str, split: int):
        root = Path(root)
        if split == 'train':
            files = root.rglob('*.wav')
            excludes = []
            with open(root / 'testing_list.txt') as f1, open(root / 'validation_list.txt') as f2:
                excludes += f1.read().splitlines()
                excludes += f2.read().splitlines()
            
            excludes = list(map(lambda x: str(root / x), excludes))
            files = list(filter(lambda x: "_background_noise_" not in str(x) and str(x) not in excludes, files))
        else:
            split = 'testing' if split == 'test' else 'validation'
            with open(root / f'{split}_list.txt') as f:
                files = f.read().splitlines()

            files = list(map(lambda x: root / x, files))

        targets = list(map(lambda x: self.CLASSES.index(str(x.parent).rsplit(os.path.sep, maxsplit=1)[-1]), files))
        assert len(files) == len(targets)
        return files, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        audio, _ = torchaudio.load(self.data[index]) 
        audio = self.resample(audio)
        if audio.shape[1] < 32000:  audio = torch.cat([audio, torch.zeros(1, 32000-audio.shape[1])], dim=-1)   # if less than 1s, pad the audio
        audio = self.mel_tf(audio)                          # convert to mel spectrogram    
        audio = 10.0 * audio.clamp_(1e-10).log10()          # convert to log mel spectrogram
        if self.transform: audio = self.transform(audio)
        target = int(self.targets[index])
        return audio, target


class SpeechCommandsv2(SpeechCommandsv1):
    CLASSES = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
               'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']


if __name__ == '__main__':
    dataset = SpeechCommandsv2('C:\\Users\\sithu\\Documents\\Datasets\\SpeechCommandsv2', 'train')
    dataloader = DataLoader(dataset, 4, True)
    for audio, target in dataloader: 
        print(audio.shape, target)
        print(audio.min(), audio.max())
        break
