import argparse
import yaml
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')
from models import get_model
from datasets import get_dataset
from utils.utils import setup_cudnn
from utils.metrics import accuracy


@torch.no_grad()
def evaluate(dataloader, model, device, loss_fn = None):
    print('Evaluating...')
    model.eval()
    test_loss, acc = 0.0, 0.0, 0.0

    for img, lbl in tqdm(dataloader):
        img = img.to(device)
        lbl = lbl.to(device)

        pred = model(img)
        
        if loss_fn:
            test_loss += loss_fn(pred, lbl).item()

        ac = accuracy(pred, lbl, topk=(1))[0]
        acc += ac * img.shape[0]
        
    test_loss /= len(dataloader.dataset)
    acc /= len(dataloader.dataset)

    return test_loss, 100*acc


def main(cfg):
    device = torch.device(cfg['DEVICE'])

    model = get_model(cfg['MODEL']['NAME'], cfg['DATASET']['NUM_CLASSES'])
    model.load_state_dict(torch.load(cfg['MODEL_PATH'], map_location='cpu'))
    model = model.to(device)

    _, val_dataset = get_dataset(cfg, val_transform=None)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True)

    _, acc = evaluate(val_dataloader, model, device)

    print(f"Accuracy: {acc}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/defaults.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    setup_cudnn()
    main(cfg)
