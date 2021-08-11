import torch
import torchaudio
from pathlib import Path
from torch import Tensor
from torchaudio import transforms as T
from torchvision import transforms as VT
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class FSDKaggle2018(Dataset):
    """
    """
    CLASSES = ["Acoustic_guitar", "Applause", "Bark", "Bass_drum", "Burping_or_eructation", "Bus", "Cello", "Chime", "Clarinet", "Computer_keyboard", "Cough", "Cowbell", "Double_bass", "Drawer_open_or_close", "Electric_piano", "Fart", "Finger_snapping", "Fireworks", "Flute",
               "Glockenspiel", "Gong", "Gunshot_or_gunfire", "Harmonica", "Hi-hat", "Keys_jangling", "Knock", "Laughter", "Meow", "Microwave_oven", "Oboe", "Saxophone", "Scissors", "Shatter", "Snare_drum", "Squeak", "Tambourine", "Tearing", "Telephone", "Trumpet", "Violin_or_fiddle", "Writing"]

    def __init__(self, root: str, split: str = 'train', channels: int = 3, transform=None) -> None:
        super().__init__()
        assert split in ['train', 'test']
        self.num_classes = len(self.CLASSES)
        self.transform = transform
        sample_rate = 44100
        n_fft = 4410
        n_mels = 128        # frequency bins
        window_sizes = [25, 50, 100]
        hop_sizes = [10, 25, 50]
        size = (128, 250)
        self.data = [1, 2]
        self.mel_transforms = [
            T.MelSpectrogram(sample_rate, n_fft, round(window_sizes[i]*sample_rate/1000), round(hop_sizes[i]*sample_rate/1000), n_mels=n_mels)
        for i in range(channels)]

        self.resize = VT.Resize(size)

        self.data, self.targets = self.get_data(root, split)

        print(f"Found {len(self.data)} audios in {root}.")

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
        if self.transform: audio = self.transform(audio)
        audio = [self.resize(mel_tf(audio).log()) for mel_tf in self.mel_transforms]
        audio = torch.cat(audio, dim=0)

        target = int(self.targets[index])
        return audio, target


if __name__ == '__main__':
    transform = None
    dataset = FSDKaggle2018('C:\\Users\\sithu\\Documents\\Datasets\\FSDKaggle2018', transform=transform)
    dataloader = DataLoader(dataset, 4, True)
    audio, target = next(iter(dataloader))
    print(audio.shape, target)
