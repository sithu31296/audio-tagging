import argparse
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')
from models import get_model
from datasets import get_val_dataset
from utils.utils import setup_cudnn
from utils.metrics import accuracy


@torch.no_grad()
def evaluate(dataloader, model, device, loss_fn = None):
    print('Evaluating...')
    model.eval()
    test_loss, acc = 0.0, 0.0

    for audio, target in tqdm(dataloader):
        audio = audio.to(device)
        target = target.to(device)

        pred = model(audio)
        
        if loss_fn:
            test_loss += loss_fn(pred, target).item()

        ac = accuracy(pred, target)[0]
        acc += ac * audio.shape[0]
        
    test_loss /= len(dataloader.dataset)
    acc /= len(dataloader.dataset)

    return test_loss, acc


def main(cfg):
    print(f"Using {cfg['DEVICE']}...")
    device = torch.device(cfg['DEVICE'])

    dataset = get_val_dataset(cfg)
    dataloader = DataLoader(dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)

    print(f"Loading Model and trained weights from {cfg['MODEL_PATH']}")
    model = get_model(cfg['MODEL']['NAME'], dataset.num_classes)
    model.load_state_dict(torch.load(cfg['MODEL_PATH'], map_location='cpu'))
    model = model.to(device)

    acc = evaluate(dataloader, model, device)[-1]

    print(f"Accuracy: {acc:.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Experiment configuration file name')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    setup_cudnn()
    main(cfg)
