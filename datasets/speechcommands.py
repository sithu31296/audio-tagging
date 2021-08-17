import torch
import random
import torchaudio
import os
from pathlib import Path
from torch import Tensor
from torchaudio import transforms as T
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from .transforms import mixup_augment


class SpeechCommandsv1(Dataset):
    CLASSES = ['bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'four', 'go', 'happy', 'house', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'wow', 'yes', 'zero']

    def __init__(self, split, data_cfg, mixup_cfg=None, transform=None, spec_transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val', 'test']
        self.num_classes = len(self.CLASSES)
        self.transform = transform
        self.spec_transform = spec_transform
        self.mixup = mixup_cfg['MIXUP'] if mixup_cfg is not None else 0.0
        self.mixup_alpha = mixup_cfg['MIXUP_ALPHA'] if mixup_cfg is not None else 0.0
        self.label_smooth = mixup_cfg['SMOOTHING'] if mixup_cfg is not None else 0.0
        self.num_frames = data_cfg['SAMPLE_RATE'] * data_cfg['AUDIO_LENGTH']

        self.mel_tf = T.MelSpectrogram(data_cfg['SAMPLE_RATE'], data_cfg['WIN_LENGTH'], data_cfg['WIN_LENGTH'], data_cfg['HOP_LENGTH'], data_cfg['FMIN'], data_cfg['FMAX'], n_mels=data_cfg['N_MELS'], norm='slaney')    # using mel_scale='slaney' is better
        self.resample = T.Resample(data_cfg['SOURCE_SAMPLE'], data_cfg['SAMPLE_RATE'])

        self.data, self.targets = self.get_data(data_cfg['ROOT'], split)
        print(f"Found {len(self.data)} {split} audios in {data_cfg['ROOT']}.")


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

        targets = list(map(lambda x: int(self.CLASSES.index(str(x.parent).rsplit(os.path.sep, maxsplit=1)[-1])), files))
        assert len(files) == len(targets)
        return files, targets


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        audio, _ = torchaudio.load(self.data[index]) 
        audio = self.resample(audio)
        if audio.shape[1] < self.num_frames:  audio = torch.cat([audio, torch.zeros(1, self.num_frames-audio.shape[1])], dim=-1)   # if less than 1s, pad the audio
        target = torch.tensor(self.targets[index])

        if self.transform: audio = self.transform(audio)

        if random.random() < self.mixup:
            next_index = random.randint(0, len(self.data)-1)
            next_audio, _ = torchaudio.load(self.data[next_index])
            next_audio = self.resample(next_audio)
            if next_audio.shape[1] < self.num_frames: next_audio = torch.cat([next_audio, torch.zeros(1, self.num_frames-next_audio.shape[1])], dim=-1)   # if less than 1s, pad the audio
            next_target = torch.tensor(self.targets[next_index])
            audio, target = mixup_augment(audio, target, next_audio, next_target, self.mixup_alpha, self.num_classes, self.label_smooth)
        else:
            target = F.one_hot(target, self.num_classes).float()
        
        audio = self.mel_tf(audio)                          # convert to mel spectrogram    
        audio = 10.0 * audio.clamp_(1e-10).log10()          # convert to log mel spectrogram
        
        if self.spec_transform: audio = self.spec_transform(audio)
        
        return audio, target


class SpeechCommandsv2(SpeechCommandsv1):
    CLASSES = ['backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
               'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']


if __name__ == '__main__':
    data_cfg = {
        'ROOT': 'C:/Users/sithu/Documents/Datasets/SpeechCommands',
        'SOURCE_SAMPLE': 16000,
        'SAMPLE_RATE': 32000,
        'AUDIO_LENGTH': 1,
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
    dataset = SpeechCommandsv1('train', data_cfg, aug_cfg)
    dataloader = DataLoader(dataset, 2, True)
    for audio, target in dataloader:
        print(audio.shape, target.argmax(dim=1))
        print(audio.min(), audio.max())
        break