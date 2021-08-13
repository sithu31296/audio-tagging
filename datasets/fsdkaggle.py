import torch
import torchaudio
from pathlib import Path
from torch import Tensor
from torchaudio import transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class FSDKaggle2018(Dataset):
    CLASSES = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute",
               "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]

    def __init__(self, root: str, split: str = 'train', sample_rate: int = 32000, win_length: int = 1024, n_mels: int = 64, fmin: int = 50, fmax: int = 14000, transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        split = 'train' if split == 'train' else 'test'
        self.num_classes = len(self.CLASSES)
        self.transform = transform

        self.mel_tf = T.MelSpectrogram(sample_rate, win_length, win_length, sample_rate//100, fmin, fmax, n_mels=n_mels, norm='slaney', mel_scale='slaney')
        self.resample = T.Resample(44100, sample_rate)
        self.data, self.targets = self.get_data(root, split)

        print(f"Found {len(self.data)} {split} audios in {root}.")

    def get_data(self, root: str, split: str):
        root = Path(root)
        csv_path = 'train_post_competition.csv' if split == 'train' else 'test_post_competition_scoring_clips.csv'
        files, targets = [], []

        with open(root / 'FSDKaggle2018.meta' / csv_path) as f:
            lines = f.read().splitlines()[1:]
            for line in lines:
                fname, label, *_ = line.split(',')
                files.append(str(root / f"audio_{split}" / fname))
                targets.append(self.CLASSES.index(label))
        return files, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        audio, _ = torchaudio.load(self.data[index])
        audio = self.resample(audio)                    # resample to 32kHz
        if audio.shape[1] < 160000:                     # if less than 5s, pad the audio
            audio = torch.cat([audio, torch.zeros(1, 160000-audio.shape[1])], dim=-1)
        else:                                           # if not, trim the audio to 5s
            audio = audio[:, :160000]
        audio = self.mel_tf(audio)                      # convert to mel spectrogram 
        audio = 10.0 * audio.clamp_(1e-10).log10()          # convert to log mel spectrogram
        if self.transform: audio = self.transform(audio)
        target = int(self.targets[index])
        return audio, target


if __name__ == '__main__':
    dataset = FSDKaggle2018('C:\\Users\\sithu\\Documents\\Datasets\\FSDKaggle2018', 'train')
    dataloader = DataLoader(dataset, 4, True)
    for audio, target in dataloader: 
        print(audio.shape, target)
        print(audio.min(), audio.max())
        break

