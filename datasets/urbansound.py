import torch
import torchaudio
from pathlib import Path
from torch import Tensor
from torchaudio import transforms as T
from torchvision import transforms as VT
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class UrbanSound8k(Dataset):
    """
    """
    CLASSES = ['air conditioner', 'car horn', 'children playing', 'dog bark', 'drilling', 'engine idling', 'gun shot', 'jackhammer', 'siren', 'street music']
    def __init__(self, root: str, fold: int = 1, channels: int = 3, transform=None) -> None:
        super().__init__()
        self.num_classes = len(self.CLASSES)
        self.transform = transform
        sample_rate = 44100
        n_fft = 4410
        n_mels = 128        # frequency bins (128 is the best)
        window_sizes = [25, 50, 100]
        hop_sizes = [10, 25, 50]
        size = (128, 250)

        self.mel_transforms = [
            T.MelSpectrogram(sample_rate, n_fft, round(window_sizes[i]*sample_rate/1000), round(hop_sizes[i]*sample_rate/1000), n_mels=n_mels)
        for i in range(channels)]

        self.resize = VT.Resize(size)

        self.data, self.targets = self.get_data(root, fold)

        print(f"Found {len(self.data)} audios in {root}.")

    def get_data(self, root: str, fold: int):
        root = Path(root)
        files = (root / 'audio').glob('*.wav')
        files = list(filter(lambda x: x.stem.startswith(f"{fold}"), files))
        targets = list(map(lambda x: x.stem.rsplit('-', maxsplit=1)[-1], files))
        assert len(files) == len(targets)
        return files, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        audio, _ = torchaudio.load(self.data[index])        # [1, 220500] >> [1, 128, 501] >> [1, 128, 500]
        if self.transform: audio = self.transform(audio)
        audio = [self.resize(mel_tf(audio).log()) for mel_tf in self.mel_transforms]
        audio = torch.cat(audio, dim=0)

        target = int(self.targets[index])
        return audio, target


if __name__ == '__main__':
    transform = None
    dataset = ESC50('C:\\Users\\sithu\\Documents\\Datasets\\ESC50', transform=transform)
    dataloader = DataLoader(dataset, 2, True)
    audio, target = next(iter(dataloader))
    print(audio.shape, target)
