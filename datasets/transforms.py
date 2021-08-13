import random
import librosa
import torch
from torch import nn, Tensor
from torchaudio import transforms as T
from torchvision import transforms as VT
from torchaudio import functional as TF
from typing import Tuple



class MixUp:
    def __init__(self, alpha: float = 32.0) -> None:
        self.alpha = alpha

    def __call__(self, audio: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        audio = self.alpha * audio[::2].transpose(0, -1) + (1 - self.alpha) * audio[1::2].transpose(0, -1)
        return audio.transpose(0, -1), target


class DropStripes(nn.Module):
    def __init__(self, dim: int, drop_width: int, num_stripes: int) -> None:
        super().__init__()
        assert dim in [2, 3]    # dim-2 = time, dim-3 = frequency
        self.dim = dim
        self.drop_width = drop_width
        self.num_stripes = num_stripes

    def transform_slice(self, e: Tensor, total_width: int):
        # e (channels, time_steps, freq_bins)
        for _ in range(self.num_stripes):
            distance = torch.randint(0, self.drop_width, size=(1,))[0]
            bgn = torch.randint(0, total_width-distance, size=(1,))[0]

            if self.dim == 2:
                e[:, bgn:bgn+distance, :] = 0
            else:
                e[:, :, bgn:bgn+distance] = 0


    def forward(self, x: Tensor) -> Tensor:
        # x [B, C, time_steps, freq_bins]
        if self.training:
            for y in x: self.transform_slice(y, x.shape[self.dim])
        return x


class SpecAug(nn.Module):
    """Spec Augmentation
    Park, D.S., Chan, W., Zhang, Y., Chiu, C.C., Zoph, B., Cubuk, E.D. 
    and Le, Q.V., 2019. Specaugment: A simple data augmentation method 
    for automatic speech recognition. arXiv preprint arXiv:1904.08779.
    """
    def __init__(self, time_drop_width: int, time_num_stripes: int, freq_drop_width: int, freq_num_stripes: int):
        super().__init__()
        self.time_dropper = DropStripes(2, time_drop_width, time_num_stripes)
        self.freq_dropper = DropStripes(3, freq_drop_width, freq_num_stripes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.time_dropper(x)
        print(x.sum())
        x = self.freq_dropper(x)
        print(x.sum())
        return x



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
        new_audio = librosa.effects.time_stretch(new_audio, time_stretch, )
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



if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.randn(4, 1, 64, 701)
    aug = MixUp()
    y, _ = aug(x, torch.tensor([4, 3, 5, 6]))
    print(y.shape)