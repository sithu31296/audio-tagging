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
from utils.visualize import plot_sound_events


class SED:
    def __init__(self, cfg) -> None:
        self.device = torch.device(cfg['DEVICE'])
        self.labels = np.array(__all__[cfg['DATASET']['NAME']].CLASSES)
        self.model = get_model(cfg['MODEL']['NAME'], len(self.labels))
        self.model.load_state_dict(torch.load(cfg['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        self.threshold = cfg['TEST']['THRESHOLD']
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
        probs, indices = torch.sort(prob.max(dim=0)[0], descending=True)
        indices = indices[probs > self.threshold]
        # probs, indices = torch.topk(prob.max(dim=0)[0], topk)
        top_results = prob[:, indices].t()

        starts = []
        ends = []

        for result in top_results:
            index = torch.where(result > self.threshold)[0]
            starts.append(round(index[0].item()/100, 1))
            ends.append(round(index[-1].item()/100, 1))

        return top_results, self.labels[indices], starts, ends
            

    @torch.no_grad()
    def model_forward(self, audio: Tensor) -> Tensor:
        start = time_sync()
        pred = self.model(audio)[0].squeeze().cpu()
        end = time_sync()
        print(f"PyTorch Model Inference Time: {(end-start)*1000:.2f}ms")
        return pred

    def predict(self, file: str) -> str:
        audio = self.preprocess(file)
        pred = self.model_forward(audio)
        results, labels, starts, ends = self.postprocess(pred)
        return results, labels, starts, ends

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/sed.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    file_path = Path(cfg['TEST']['FILE'])
    model = SED(cfg)

    if cfg['TEST']['MODE'] == 'file':
        if file_path.is_file():
            results, labels, starts, ends = model.predict(str(file_path))
            print(tabulate({"Class": labels, "Start": starts, "End": ends}, headers='keys'))
            if cfg['TEST']['PLOT']:
                plot_sound_events(results, labels)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError