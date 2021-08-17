import argparse
import torch
import yaml
import time
import multiprocessing as mp
from pprint import pprint
from tqdm import tqdm
from tabulate import tabulate
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(0, '.')
from datasets import get_train_dataset, get_val_dataset, get_sampler
from datasets.transforms import get_waveform_transforms, get_spec_transforms
from models import get_model
from utils.utils import fix_seeds, time_sync, setup_cudnn, setup_ddp, cleanup_ddp
from utils.schedulers import get_scheduler
from utils.losses import get_loss
from utils.optimizers import get_optimizer
from val import evaluate


def main(cfg, gpu, save_dir):
    start = time_sync()
    
    best_score = 0.0
    num_workers = mp.cpu_count()
    device = torch.device(cfg['DEVICE'])
    train_config = cfg['TRAIN']
    epochs = train_config['EPOCHS']
    metric = cfg['DATASET']['METRIC']
    lr = cfg['OPTIMIZER']['LR']
    
    # augmentations
    # waveform_transforms = get_waveform_transforms(cfg['DATASET']['SAMPLE_RATE'], cfg['DATASET']['AUDIO_LENGTH'], 0.1, 3, 0.005)
    waveform_transforms = None
    spec_transforms = get_spec_transforms(cfg['DATASET'], cfg['AUG'])

    # dataset
    train_dataset = get_train_dataset(cfg['DATASET'], cfg['AUG'], waveform_transforms, spec_transforms)
    val_dataset = get_val_dataset(cfg['DATASET'])

    # dataset sampler
    train_sampler, val_sampler = get_sampler(train_config['DDP'], train_dataset, val_dataset)
    
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=train_config['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=train_config['BATCH_SIZE'], num_workers=num_workers, pin_memory=True, sampler=val_sampler)
    
    # create model
    model = get_model(cfg['MODEL']['NAME'], train_dataset.num_classes)
    model._init_weights(cfg['MODEL']['PRETRAINED'])
    model = model.to(device)
    if train_config['DDP']: model = DDP(model, device_ids=[gpu])

    # loss function, optimizer, scheduler, AMP scaler, tensorboard writer
    loss_fn = get_loss(train_config['LOSS'])
    optimizer = get_optimizer(model, cfg['OPTIMIZER']['NAME'], lr, cfg['OPTIMIZER']['WEIGHT_DECAY'])
    scheduler = get_scheduler(cfg, optimizer)
    scaler = GradScaler(enabled=train_config['AMP'])
    writer = SummaryWriter(save_dir / 'logs')
    iters_per_epoch = len(train_dataset) // train_config['BATCH_SIZE']

    for epoch in range(epochs):
        model.train()
        
        if train_config['DDP']: train_sampler.set_epoch(epoch)

        train_loss = 0.0
        pbar = tqdm(enumerate(train_dataloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.6f}")
        
        for iter, (audio, target) in pbar:
            audio = audio.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            with autocast(enabled=train_config['AMP']):
                pred = model(audio)
                loss = loss_fn(pred, target)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr = scheduler.get_last_lr()[0]
            train_loss += loss.item() 

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss/(iter+1):.6f}")

        train_loss /= iter+1
        writer.add_scalar('train/loss', train_loss, epoch)

        scheduler.step()
        torch.cuda.empty_cache()

        if (epoch+1) % train_config['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            # evaluate the model
            score = evaluate(val_dataloader, model, device, metric)[0]
            writer.add_scalar(f'val/{metric}', score, epoch)

            if score >= best_score:
                best_score = score
                torch.save(model.module.state_dict() if train_config['DDP'] else model.state_dict(), save_dir / f"{cfg['MODEL']['NAME']}_{cfg['DATASET']['NAME']}.pth")
            print(f"Current {metric}: {score:.2f} Best {metric}: {best_score:.2f}")
  
    end = time.gmtime(time_sync() - start)
    total_time = time.strftime("%H:%M:%S", end)

    # results table
    table = [
        [f'Best {metric}', f"{best_score:.2f}"],
        ['Total Training Time', total_time]
    ]
    print(tabulate(table, numalign='right'))

    writer.close()
    pbar.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Experiment configuration file name')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    pprint(cfg)
    save_dir = Path(cfg['TRAIN']['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    fix_seeds(123)
    setup_cudnn()
    gpu = setup_ddp()
    main(cfg, gpu, save_dir)
    cleanup_ddp()