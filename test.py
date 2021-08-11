# from IPython.display import Audio
import torchaudio
import matplotlib.pyplot as plt

# FSDKaggle2018 parameters
clip_length = 1.0
sample_rate = 44100
hop_length = 441
n_fft = 1024
n_mels = 64
f_min = 0
f_max = 22050


audio, sr = torchaudio.load('file.wav')
plt.plot(audio.t().numpy())
# Audio(audio[0, :sr], rate=sr)