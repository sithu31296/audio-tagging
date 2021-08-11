import torch
import argparse
import torchaudio
import yaml
import numpy as np
from torch import Tensor
from tabulate import tabulate
from pathlib import Path
from torchaudio import transforms as T
from torchaudio import functional as F

import sys
sys.path.insert(0, '.')
from models import get_model
from datasets import __all__
from utils.utils import time_sync


class AudioTagging:
    def __init__(self, cfg) -> None:
        self.device = torch.device(cfg['DEVICE'])
        self.labels = np.array(__all__[cfg['DATASET']['NAME']].CLASSES)
        self.model = get_model(cfg['MODEL']['NAME'], len(self.labels))
        self.model.load_state_dict(torch.load(cfg['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.topk = cfg['TEST']['TOPK']
        self.sample_rate = cfg['SAMPLE_RATE']
        self.mel_tf = T.MelSpectrogram(cfg['SAMPLE_RATE'], cfg['WIN_LENGTH'], cfg['WIN_LENGTH'], cfg['SAMPLE_RATE']//100, cfg['FMIN'], cfg['FMAX'], n_mels=cfg['N_MELS'], norm='slaney', mel_scale='slaney')

    def preprocess(self, file: str) -> Tensor:
        audio, sr = torchaudio.load(file)
        if sr != self.sample_rate: audio = F.resample(audio, sr, self.sample_rate, dtype=audio.dtype)
        audio = self.mel_tf(audio).log10()
        audio *= 10.0
        audio = audio.unsqueeze(0)
        audio = audio.to(self.device)
        return audio

    def postprocess(self, prob: Tensor) -> str:
        probs, indices = torch.topk(prob.squeeze().cpu(), self.topk)
        return self.labels[indices], probs

    @torch.no_grad()
    def model_forward(self, audio: Tensor) -> Tensor:
        start = time_sync()
        pred = self.model(audio)
        end = time_sync()
        print(f"PyTorch Model Inference Time: {(end-start)*1000:.2f}ms")
        return pred

    def predict(self, file: str) -> str:
        audio = self.preprocess(file)
        pred = self.model_forward(audio)
        labels, probs = self.postprocess(pred)
        return labels, probs

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/tagging.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    file_path = Path(cfg['TEST']['FILE'])
    model = AudioTagging(cfg)

    if cfg['TEST']['MODE'] == 'file':
        if file_path.is_file():
            labels, probs = model.predict(str(file_path))
            print(tabulate({"Class": labels, "Probability": probs}, headers='keys'))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError