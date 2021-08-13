import torchaudio
from pathlib import Path
from torch import Tensor
from torchaudio import transforms as T
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class ESC50(Dataset):
    """
    50 classes
    40 examples per class
    2000 recordings
    5 major categories: [Animals, Nature sounds, Human non-speech sounds, Interior/domestic sounds, Exterior/urban sounds]
    Each of the audio is named like this:
        {FOLD}-{CLIP_ID}-{TAKE}-{TARGET}.wav
    """
    CLASSES = ['dog', 'rooster', 'pig', 'cow', 'frog', 'cat', 'hen', 'insects', 'sheep', 'crow', 'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds', 'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm', 'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing', 'footsteps', 'laughing', 'brushing_teeth',
               'snoring', 'drinking_sipping', 'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening', 'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking', 'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine', 'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw']

    def __init__(self, root: str, split: str = 'train', sample_rate: int = 32000, win_length: int = 1024, n_mels: int = 64, fmin: int = 0, fmax: int = None, transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.num_classes = len(self.CLASSES)
        self.transform = transform
        val_fold = 5

        self.mel_tf = T.MelSpectrogram(sample_rate, win_length, win_length, sample_rate//100, fmin, fmax, n_mels=n_mels, norm='slaney', mel_scale='slaney')
        self.resample = T.Resample(44100, sample_rate)

        self.data, self.targets = self.get_data(root, split, val_fold)

        print(f"Found {len(self.data)} {split} audios in {root}.")

    def get_data(self, root: str, split: int, fold: int):
        root = Path(root)
        files = (root / 'audio').glob('*.wav')
        files = list(filter(lambda x: not x.stem.startswith(f"{fold}") if split == 'train' else x.stem.startswith(f"{fold}"), files))
        targets = list(map(lambda x: x.stem.rsplit('-', maxsplit=1)[-1], files))
        assert len(files) == len(targets)
        return files, targets

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        audio, _ = torchaudio.load(self.data[index])    
        audio = self.resample(audio)
        audio = self.mel_tf(audio)   
        audio = 10.0 * audio.clamp_(1e-10).log10()
        if self.transform: audio = self.transform(audio)
        target = int(self.targets[index])
        return audio, target


if __name__ == '__main__':
    transform = None
    dataset = ESC50('C:\\Users\\sithu\\Documents\\Datasets\\ESC50', transform=transform)
    dataloader = DataLoader(dataset, 4)
    audio, target = next(iter(dataloader))
    print(audio.shape, target)
    print(audio.min(), audio.max())
