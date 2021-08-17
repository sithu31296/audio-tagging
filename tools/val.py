import argparse
import yaml
import torch
import multiprocessing as mp
from pathlib import Path
from pprint import pprint
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, '.')
from models import get_model
from datasets import get_val_dataset
from utils.utils import setup_cudnn
from utils.metrics import Metrics


@torch.no_grad()
def evaluate(dataloader, model, device, metric='accuracy'):
    print('Evaluating...')
    model.eval()
    ametrics = Metrics(metric)

    for audio, target in tqdm(dataloader, ):
        audio = audio.to(device)
        target = target.to(device)
        pred = model(audio)  
        ametrics.update(pred, target)  

    return ametrics.compute()


def main(cfg):
    save_dir = Path(cfg['TRAIN']['SAVE_DIR'])
    device = torch.device(cfg['DEVICE'])
    metric_name = cfg['DATASET']['METRIC']
    num_workers = mp.cpu_count() 

    dataset = get_val_dataset(cfg['DATASET'])
    dataloader = DataLoader(dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], num_workers=num_workers, pin_memory=True)
    model = get_model(cfg['MODEL']['NAME'], dataset.num_classes)

    try:
        model_weights = save_dir / f"{cfg['MODEL']['NAME']}_{cfg['DATASET']['NAME']}.pth"
        model.load_state_dict(torch.load(str(model_weights), map_location='cpu'))
        print(f"Loading Model and trained weights from {model_weights}")
    except:
        print(f"Please consider placing your model's weights in {save_dir}")
    
    model = model.to(device)

    if metric_name == 'accuracy':
        acc = evaluate(dataloader, model, device, metric_name)[-1]
        table = [['Accuracy', f"{acc:.2f}"]]
    else:
        mAP, mAUC, d_prime = evaluate(dataloader, model, device, metric_name)
        table = [
            ['mAP', f"{mAP:.2f}"],
            ['AUC', f"{mAUC:.2f}"],
            ['d-prime', f"{d_prime:.2f}"]
        ]
    print(tabulate(table, numalign='right'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Experiment configuration file name')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    pprint(cfg)
    setup_cudnn()
    main(cfg)