import torch
import librosa
import argparse
import argparse
import pandas as pd
import pickle
from pathlib import Path
from torchaudio import transforms as T
from torchvision import transforms as VT


def extract_mel(metas, root, sr):
    files = list(metas.filename.unique())
    values = []

    num_channels = 3
    window_sizes = [25, 50, 100]
    hop_sizes = [10, 25, 50]
    n_fft = 4410
    n_mels = 128
    eps = 1e-6
    size = (128, 250)

    for f in files:
        audio, _ = librosa.load(root / 'audio' / f, sr=sr)
        audio = torch.tensor(audio)
        
        entries = metas.loc[metas['filename'] == f].to_dict(orient='records')

        for data in entries:
            specs = []

            for i in range(num_channels):
                window_length = round(window_sizes[i] * sr/1000)
                hop_length = round(hop_sizes[i] * sr/1000)

                spec = T.MelSpectrogram(sr, n_fft, window_length, hop_length, n_mels=n_mels)(audio)
                spec = torch.log(spec+eps).unsqueeze(0)
                spec = VT.Resize(size)(spec)
                specs.append(spec)

            values.append({
                "audio": torch.cat(specs, dim=0),
                "target": torch.tensor(data['target'])
            })
    return values


def main(root, store, sr):
    root = Path(root)
    store = Path(store)
    store.mkdir(exist_ok=True)

    metas = pd.read_csv(root / 'meta' / 'esc50.csv', skipinitialspace=True)

    num_folds = 5


    for i in range(1, num_folds+1):
        train_set = metas.loc[metas['fold'] != i]
        val_set = metas.loc[metas['fold'] == i]
        
        train_mel = extract_mel(train_set, root, sr)
        val_mel = extract_mel(val_set, root, sr)

        print('Start saving...')
        with open(store / f"training128mel{i}.pkl", "wb") as f:
            pickle.dump(train_mel, f)

        with open(store / f"validation128mel{i}.pkl", "wb") as f:
            pickle.dump(val_mel, f)
        break



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default='C:\\Users\\sithu\\Documents\\Datasets\\ESC50')
    parser.add_argument('--store-dir', type=str, default='C:\\Users\\sithu\\Documents\\Datasets\\ESC50\\mel')
    parser.add_argument('--sr', type=int, default=44100)
    args = parser.parse_args()

    main(args.dataset_dir, args.store_dir, args.sr)