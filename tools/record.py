import torch
import sys
import torchaudio
import queue
import threading
import sounddevice as sd
import numpy as np

from torch import Tensor



def record_audio(duration: int, sample_rate: int, channels: int = 1):
    frames = int(duration * sample_rate)
    dtype = 'float32'
    audio = sd.rec(frames, sample_rate, channels, dtype)      # records in the background
    sd.wait()

def show_devices():
    print(sd.query_devices())