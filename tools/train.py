import argparse
import torch
import yaml
import time
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
from models import get_model
from utils.utils import fix_seeds, time_sync, setup_cudnn, setup_ddp, cleanup_ddp
from utils.schedulers import get_scheduler
from utils.losses import get_loss
from utils.optimizers import get_optimizer
from val import evaluate


def main(cfg):
    start = time_sync()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)

    print(f"Using {cfg['DEVICE']}...")
    device = torch.device(cfg['DEVICE'])
    ddp_enable = cfg['TRAIN']['DDP']
    epochs = cfg['TRAIN']['EPOCHS']
    best_acc = 0.0
    if ddp_enable: gpu = setup_ddp()

    # dataset
    train_dataset = get_train_dataset(cfg, None)
    val_dataset = get_val_dataset(cfg)

    # dataset sampler
    train_sampler, val_sampler = get_sampler(cfg, train_dataset, val_dataset)
    
    # dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['TRAIN']['BATCH_SIZE'], num_workers=cfg['TRAIN']['WORKERS'], drop_last=True, pin_memory=True, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['EVAL']['BATCH_SIZE'], num_workers=cfg['EVAL']['WORKERS'], pin_memory=True, sampler=val_sampler)
    
    # training model
    model = get_model(cfg['MODEL']['NAME'], train_dataset.num_classes)
    model._init_weights(cfg['MODEL']['PRETRAINED'])
    model = model.to(device)

    if ddp_enable:
        model = DDP(model, device_ids=[gpu])

    # loss function, optimizer, scheduler, AMP scaler, tensorboard writer
    loss_fn = get_loss(cfg['TRAIN']['LOSS'])
    optimizer = get_optimizer(model, cfg['TRAIN']['OPTIMIZER']['NAME'], cfg['TRAIN']['OPTIMIZER']['LR'], cfg['TRAIN']['OPTIMIZER']['WEIGHT_DECAY'])
    scheduler = get_scheduler(cfg, optimizer)
    scaler = GradScaler(enabled=cfg['TRAIN']['AMP'])
    writer = SummaryWriter(save_dir / 'logs')
    iters_per_epoch = len(train_dataset) // cfg['TRAIN']['BATCH_SIZE']

    for epoch in range(epochs):
        model.train()
        
        if ddp_enable: train_sampler.set_epoch(epoch)
        train_loss = 0.0
        pbar = tqdm(enumerate(train_dataloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {cfg['TRAIN']['OPTIMIZER']['LR']:.8f} Loss: {0:.8f}")
        
        for iter, (audio, target) in pbar:
            audio = audio.to(device)
            target = target.to(device)

            optimizer.zero_grad()

            with autocast(enabled=cfg['TRAIN']['AMP']):
                pred = model(audio)
                loss = loss_fn(pred, target)

            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr = scheduler.get_last_lr()[0]
            train_loss += loss.item() 

            pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {loss.item():.8f}")

        train_loss /= len(train_dataset)
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/lr', lr, epoch)
        writer.flush()

        scheduler.step()
        torch.cuda.empty_cache()

        if (epoch % cfg['TRAIN']['EVAL_INTERVAL'] == 0) and (epoch >= cfg['TRAIN']['EVAL_INTERVAL']):
            # evaluate the model
            test_loss, acc = evaluate(val_dataloader, model, device, loss_fn)

            writer.add_scalar('val/loss', test_loss, epoch)
            writer.add_scalar('val/Acc', acc, epoch)
            writer.flush()

            if acc > best_acc:
                best_acc = acc
                torch.save(model.module.state_dict() if ddp_enable else model.state_dict(), save_dir / f"{cfg['MODEL']['NAME']}_{cfg['DATASET']['NAME']}.pth")
            print(f"Validation Loss: {test_loss:>8f} Current Accuracy: {acc:.2f} Best Accuracy: {best_acc:.2f}")
        
    writer.close()
    pbar.close()
  
    end = time.gmtime(time_sync() - start)
    total_time = time.strftime("%H:%M:%S", end)

    # results table
    table = [
        ['Best Accuracy', f"{best_acc:.2f}"],
        ['Total Training Time', total_time]
    ]
      
    print(tabulate(table, numalign='right'))

    if ddp_enable: cleanup_ddp()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True, help='Experiment configuration file name')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    fix_seeds(123)
    setup_cudnn()
    main(cfg)