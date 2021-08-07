import random
import librosa
import torch
from torch import Tensor
from torchaudio import transforms as T
from torchvision import transforms as VT

import pickle
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class AudioAug:
    def __init__(self, bins, mode) -> None:
        self.window_length = [25, 50, 100]
        self.hop_length = [10, 25, 50]
        self.fft = 4410
        self.bins = bins
        self.mode = mode
        self.sr = 44100
        self.length = 250
        self.eps = 1e-6

    def __call__(self, audio: Tensor) -> Tensor:
        limits = ((-2, 2), (0.9, 1.2))

        pitch_shift = random.randint(limits[0][0], limits[0][1]+1)
        time_stretch = random.random() * (limits[1][1] - limits[1][0]) + limits[1][0]
        new_audio = librosa.effects.pitch_shift(audio, self.sr, pitch_shift)
        new_audio = librosa.effects.time_stretch(new_audio, time_stretch)
        audio = torch.tensor(new_audio)

        specs = []

        for i in range(len(self.window_length)):
            window_length = round(window_length[i] * self.sr/1000)
            hop_length = round(hop_length[i] * self.sr/1000)
            spec = T.MelSpectrogram(self.sr, self.fft, window_length, hop_length, n_mels=self.bins)(audio)
            spec = torch.log(spec+self.eps).unsqueeze(0)
            spec = VT.Resize((128, self.length))(spec)
            specs.append(spec)
        return torch.cat(specs, dim=0)




class ESC50(Dataset):
    def __init__(self, root: str, transform=None) -> None:
        super().__init__()
        self.num_classes = 50
        self.transform = transform
        with open(root, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Found {len(self.data)} audios in {root}.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        data = self.data[index]
        audio, target = data['audio'], data['target']

        if self.transform:
            audio = self.transform(audio)
        return audio, target.long()


if __name__ == '__main__':
    dataset = ESC50('C:\\Users\\sithu\\Documents\\Datasets\\ESC50\\mel\\validation128mel1.pkl')
    dataloader = DataLoader(dataset, 4, True)
    audio, target = next(iter(dataloader))
    print(audio.shape, target)