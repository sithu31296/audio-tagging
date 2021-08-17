import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor
from torchaudio import functional as AF


def plot_waveform(waveform: Tensor, sample_rate: int):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    fig, axes = plt.subplots(num_channels, 1)
    if num_channels == 1: axes = [axes]

    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1: axes[c].set_ylabel(f"Channel {c+1}")
        
    fig.suptitle("Waveform")
    plt.show(block=False)

def plot_specgram(waveform: Tensor, sample_rate: int):
    waveform = waveform.numpy()
    num_channels, _ = waveform.shape

    fig, axes = plt.subplots(num_channels, 1)
    if num_channels == 1: axes = [axes]

    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1: axes[c].set_ylabel(f"Channel {c+1}")
        
    fig.suptitle("Spectrogram")
    plt.show(block=False)

def plot_spectrogram(spec: Tensor):
    fig, ax = plt.subplots(1, 1)
    ax.set_title("Spectrogram (db)")
    ax.set_ylabel('freq_bin')
    ax.set_xlabel('frame')
    im = ax.imshow(AF.amplitude_to_DB(spec[0], 10, 1e-10, np.log10(max(spec.max(), 1e-10))).numpy(), origin='lower', aspect='auto')
    fig.colorbar(im, ax=ax)
    plt.show(block=False)

def plot_mel_fbank(fbank: Tensor):
    plt.imshow(fbank.numpy(), aspect='auto')
    plt.xlabel('mel_bin')
    plt.ylabel('freq_bin')
    plt.title('Filter bank')
    plt.show(block=False)

def plot_pitch(waveform: Tensor, sample_rate: int, pitch: Tensor):
    num_channels, num_frames = waveform.shape
    pitch_channels, pitch_frames = pitch.shape
    time_axis = torch.linspace(0, num_frames / sample_rate, num_frames)
    pitch_axis = torch.linspace(0, num_frames / sample_rate, pitch_frames)

    plt.plot(time_axis, num_channels, linewidth=1, color='gray', alpha=0.3, label='Waveform')
    plt.plot(pitch_axis, pitch_channels, linewidth=2, color='green', label='Pitch')
    plt.title("Pitch Feature")
    plt.grid(True)
    plt.legend(loc=0)
    plt.show(block=False)


def play_audio(waveform: Tensor, sample_rate: int):
    from IPython.display import Audio, display
    waveform = waveform.numpy()
    num_channels, _ = waveform.shape
    
    if num_channels == 1: 
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")

def plot_sound_events(results: Tensor, labels: list, fps: int = 100):
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.matshow(results, origin='upper', aspect='auto', cmap='jet', vmin=0, vmax=1)
    ax.set_title('Sound Event Detection')
    ax.set_xlabel('Seconds')
    ax.set_xticks(np.arange(0, results.shape[1], fps))
    ax.set_xticklabels(np.arange(0, results.shape[1]/fps))
    ax.set_yticks(np.arange(0, results.shape[0]))
    ax.set_yticklabels(labels)
    ax.yaxis.grid(color='k', linestyle='solid', linewidth=0.3, alpha=0.3)
    ax.xaxis.set_ticks_position('bottom')
    
    plt.tight_layout()
    plt.show()
    # plt.savefig('result.jpg')