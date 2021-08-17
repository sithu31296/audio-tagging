import torch
import random
import torchaudio
from pathlib import Path
from torch import Tensor
from torchaudio import transforms as T
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from .transforms import mixup_augment


class FSDKaggle2018(Dataset):
    CLASSES = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute",
               "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]

    def __init__(self, split, data_cfg, mixup_cfg=None, transform=None, spec_transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        split = 'train' if split == 'train' else 'test'
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


    def get_data(self, root: str, split: str):
        root = Path(root)
        csv_path = 'train_post_competition.csv' if split == 'train' else 'test_post_competition_scoring_clips.csv'
        files, targets = [], []

        with open(root / 'FSDKaggle2018.meta' / csv_path) as f:
            lines = f.read().splitlines()[1:]
            for line in lines:
                fname, label, *_ = line.split(',')
                files.append(str(root / f"audio_{split}" / fname))
                targets.append(int(self.CLASSES.index(label)))
        return files, targets


    def __len__(self) -> int:
        return len(self.data)


    def cut_pad(self, audio: Tensor) -> Tensor:
        if audio.shape[1] < self.num_frames:                     # if less than 5s, pad the audio
            audio = torch.cat([audio, torch.zeros(1, self.num_frames-audio.shape[1])], dim=-1)
        else:                                           # if not, trim the audio to 5s
            audio = audio[:, :self.num_frames]
        return audio


    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        audio, _ = torchaudio.load(self.data[index])
        audio = self.resample(audio)                    # resample to 32kHz
        audio = self.cut_pad(audio)
        target = torch.tensor(self.targets[index])

        if self.transform: audio = self.transform(audio)

        if random.random() < self.mixup:
            next_index = random.randint(0, len(self.data)-1)
            next_audio, _ = torchaudio.load(self.data[next_index])
            next_audio = self.resample(next_audio)
            next_audio = self.cut_pad(next_audio)
            next_target = torch.tensor(self.targets[next_index])
            audio, target = mixup_augment(audio, target, next_audio, next_target, self.mixup_alpha, self.num_classes, self.label_smooth)
        else:
            target = F.one_hot(target, self.num_classes).float()

        audio = self.mel_tf(audio)                      # convert to mel spectrogram 
        audio = 10.0 * audio.clamp_(1e-10).log10()          # convert to log mel spectrogram
        
        if self.spec_transform: audio = self.spec_transform(audio)
        
        return audio, target


if __name__ == '__main__':
    data_cfg = {
        'ROOT': 'C:/Users/sithu/Documents/Datasets/FSDKaggle2018',
        'SOURCE_SAMPLE': 44100,
        'SAMPLE_RATE': 32000,
        'AUDIO_LENGTH': 5,
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
    dataset = FSDKaggle2018('val', data_cfg, aug_cfg)
    dataloader = DataLoader(dataset, 2, True)
    for audio, target in dataloader:
        print(audio.shape, target.argmax(dim=1))
        print(audio.min(), audio.max())
        break

