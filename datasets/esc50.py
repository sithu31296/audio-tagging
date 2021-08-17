import torch
import random
import torchaudio
from pathlib import Path
from torch import Tensor
from torchaudio import transforms as T
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple
from .transforms import mixup_augment


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

    def __init__(self, split, data_cfg, mixup_cfg=None, transform=None, spec_transform=None) -> None:
        super().__init__()
        assert split in ['train', 'val']
        self.num_classes = len(self.CLASSES)
        self.transform = transform
        self.spec_transform = spec_transform
        self.mixup = mixup_cfg['MIXUP'] if mixup_cfg is not None else 0.0
        self.mixup_alpha = mixup_cfg['MIXUP_ALPHA'] if mixup_cfg is not None else 0.0
        self.label_smooth = mixup_cfg['SMOOTHING'] if mixup_cfg is not None else 0.0
        val_fold = 5

        self.mel_tf = T.MelSpectrogram(data_cfg['SAMPLE_RATE'], data_cfg['WIN_LENGTH'], data_cfg['WIN_LENGTH'], data_cfg['HOP_LENGTH'], data_cfg['FMIN'], data_cfg['FMAX'], n_mels=data_cfg['N_MELS'], norm='slaney')    # using mel_scale='slaney' is better
        self.resample = T.Resample(data_cfg['SOURCE_SAMPLE'], data_cfg['SAMPLE_RATE'])

        self.data, self.targets = self.get_data(data_cfg['ROOT'], split, val_fold)
        print(f"Found {len(self.data)} {split} audios in {data_cfg['ROOT']}.")


    def get_data(self, root: str, split: str, fold: int):
        root = Path(root)
        files = (root / 'audio').glob('*.wav')
        files = list(filter(lambda x: not x.stem.startswith(f"{fold}") if split == 'train' else x.stem.startswith(f"{fold}"), files))
        targets = list(map(lambda x: int(x.stem.rsplit('-', maxsplit=1)[-1]), files))
        assert len(files) == len(targets)
        return files, targets


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        audio, _ = torchaudio.load(self.data[index]) 
        audio = self.resample(audio)  
        target = torch.tensor(self.targets[index]) 

        if self.transform: audio = self.transform(audio)

        if random.random() < self.mixup:
            next_index = random.randint(0, len(self.data)-1)
            next_audio, _ = torchaudio.load(self.data[next_index])
            next_audio = self.resample(next_audio)
            next_target = torch.tensor(self.targets[next_index])
            audio, target = mixup_augment(audio, target, next_audio, next_target, self.mixup_alpha, self.num_classes, self.label_smooth)
        else:
            target = F.one_hot(target, self.num_classes).float()

        audio = self.mel_tf(audio)   
        audio = 10.0 * audio.clamp_(1e-10).log10()

        if self.spec_transform: audio = self.spec_transform(audio)

        return audio, target


if __name__ == '__main__':
    data_cfg = {
        'ROOT': 'C:/Users/sithu/Documents/Datasets/ESC50',
        'SOURCE_SAMPLE': 44100,
        'SAMPLE_RATE': 32000,
        'AUDIO_LENGTH': 5,
        'WIN_LENGTH': 1024,
        'HOP_LENGTH': 320,
        'N_MELS': 64,
        'FMIN': 50,
        'FMAX': 14000
    }
    aug_cfg = {
        'MIXUP': 0.5,
        'MIXUP_ALPHA': 10,
        'SMOOTHING': 0.1
    }
    dataset = ESC50('val', data_cfg, aug_cfg)
    dataloader = DataLoader(dataset, 2, True)
    for audio, target in dataloader:
        print(audio.shape, target.argmax(dim=1))
        print(audio.min(), audio.max())
        break
