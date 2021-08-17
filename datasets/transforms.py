import torch
import random
import math
import numpy as np
from torch import nn, Tensor
from torchaudio import transforms as T
from typing import Tuple


#######################################################
## Voice Activity Detector

class VoiceActivityDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.vad = T.Vad(
            sample_rate=16000,
            trigger_level=7,
            trigger_time=0.25,
            search_time=1.0,
            allowed_gap=0.25,
            pre_trigger_time=0.0,
            boot_time=0.35,
            noise_up_time=0.1,
            noise_down_time=0.01,
            noise_reduction_amount=1.35,
            measure_freq=20.0,
            measure_duration=None,
            measure_smooth_time=0.4
        )

    def forward(self, waveform: Tensor) -> Tensor:
        waveform = self.vad(waveform)
        return waveform


#############################################################
### WAVEFORM AUGMENTATIONS  ###

class Fade(nn.Module):
    def __init__(self, fio: float = 0.1, sample_rate: int = 32000, audio_length: int = 5):
        super().__init__()
        fiop = int(fio * sample_rate * audio_length)
        self.fade = T.Fade(fiop, fiop)

    def forward(self, waveform: Tensor) -> Tensor:
        return self.fade(waveform)


class Volume(nn.Module):
    """
    gain: in decibel (3 (weak) to 20 (strong))
    """
    def __init__(self, gain: int = 3):
        super().__init__()
        self.volume = T.Vol(gain, gain_type='db')

    def forward(self, waveform: Tensor) -> Tensor:
        return self.volume(waveform)


class GaussianNoise(nn.Module):
    def __init__(self, sigma: float = 0.005):
        super().__init__()
        self.sigma = sigma  # 0.005 (weak) to 0.02 (strong)

    def forward(self, waveform: Tensor) -> Tensor:
        gauss = np.random.normal(0, self.sigma, waveform.shape)
        waveform += gauss
        return waveform


class BackgroundNoise(nn.Module):
    """To add background noise to audio
    For simplicity, you can add audio Tensor with noise Tensor.
    A common way to adjust the intensity of noise is to change Signal-to-Noise Ratio (SNR)

        SNR = audio / noise
        SNR(db) = 10 * log10(SNR)
    """
    def __init__(self, snr_db: int = 20):
        super().__init__()
        self.snr_db = snr_db    # 3 (strong) to 20 (weak)

    def forward(self, waveform: Tensor, noise: Tensor) -> Tensor:
        # assume waveform and noise have same sample rates
        if noise.shape[1] < waveform.shape[1]:
            noise = torch.cat([noise[:1], torch.zeros(1, waveform.shape[1]-noise.shape[1])], dim=-1)
        noise = noise[:1, :waveform.shape[1]]
        scale = math.exp(self.snr_db / 10) * noise.norm(p=2) / waveform.norm(p=2)
        return (scale * waveform + noise) / 2


def mixup_augment(audio1: Tensor, target1: Tensor, audio2: Tensor, target2: Tensor, alpha: int = 10, num_classes: int = 50, smoothing: float = 0.1) -> Tuple[Tensor, Tensor]:
    ## assume audio1 and audio2 are mono channel audios and have same sampling rates
    mix_lambda = np.random.beta(alpha, alpha)

    off_value = smoothing / num_classes
    on_value = 1 - smoothing + off_value
    target1 = torch.full((1, num_classes), off_value).scatter_(1, target1.long().view(-1, 1), on_value).squeeze()
    target2 = torch.full((1, num_classes), off_value).scatter_(1, target2.long().view(-1, 1), on_value).squeeze()

    # target1 = F.one_hot(target1, num_classes)
    # target2 = F.one_hot(target2, num_classes)

    audio = audio1 * mix_lambda + audio2 * (1 - mix_lambda)
    target = target1 * mix_lambda + target2 * (1 - mix_lambda)

    return audio, target


##########################################################
## Spectrogram Augmentations

class TimeMasking(nn.Module):
    def __init__(self, mask: int = 96, audio_length: int = 5):
        super().__init__()
        assert mask < audio_length*100, f"TimeMasking parameter should be less than time frames >> {mask} > {audio_length*100}"
        self.masking = T.TimeMasking(mask)
    
    def forward(self, spec: Tensor) -> Tensor:
        return self.masking(spec)


class FrequencyMasking(nn.Module):
    def __init__(self, mask: int = 24, n_mels: int = 64):
        super().__init__()
        assert mask < n_mels, f"FrequencyMasking parameter should be less than num mels >> {mask} > {n_mels}"
        self.masking = T.FrequencyMasking(mask)

    def forward(self, spec: Tensor) -> Tensor:
        return self.masking(spec)


class FilterAugment(nn.Module):
    """
    https://github.com/frednam93/FilterAugSED
    https://arxiv.org/abs/2107.03649
    """
    def __init__(self, db_range=(-7.5, 6), n_bands=(2, 5)):
        super().__init__()
        self.db_range = db_range
        self.n_bands = n_bands

    def forward(self, audio: Tensor) -> Tensor:
        C, F, _ = audio.shape
        n_freq_band = random.randint(self.n_bands[0], self.n_bands[1])

        if n_freq_band > 1:
            band_boundary_freqs = torch.cat([
                torch.tensor([0]),
                torch.sort(torch.randint(1, F-1, (F-1,)))[0],
                torch.tensor([F])
            ])
            band_factors = torch.rand((C, n_freq_band)) * (self.db_range[1] - self.db_range[0]) + self.db_range[0]
            band_factors = 10 ** (band_factors / 20)
            freq_filter = torch.ones((C, F, 1))

            for i in range(n_freq_band):
                freq_filter[:, band_boundary_freqs[i]:band_boundary_freqs[i+1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

            audio *= freq_filter
        return audio


def get_waveform_transforms(data_config, aug_config):
    return nn.Sequential(
        Fade(0.1, data_config['SAMPLE_RATE'], data_config['AUDIO_LENGTH']),
        Volume(3),
        GaussianNoise(0.005)
    )


def get_spec_transforms(data_config, aug_config):
    return nn.Sequential(
        TimeMasking(aug_config['TIME_MASK'], data_config['AUDIO_LENGTH']),
        FrequencyMasking(aug_config['FREQ_MASK'], data_config['N_MELS'])
    )


if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.randn(4, 1, 64, 701)