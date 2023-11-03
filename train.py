import argparse
import collections
import torch
import numpy as np
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from data_loader.data_loaders import create_dataloader
from parse_config import ConfigParser
from trainer import Trainer
from writer.wandb import WandbWriter
from utils import prepare_device, load_labels, autosplit
from pathlib import Path

import wandb
import pandas as pd
import numpy as np

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    #Init dataloaders
    samples = load_labels(config['dataset']['labels_path'])
    save_split = Path(config['dataset']['labels_path']).parent
    # save_split = None
    train_samples, val_samples = autosplit(samples, test_size=config['dataset']["test_size"], save_split=save_split)

    # Trainloader
    train_loader, dataset = create_dataloader(train_samples,
                                              config['dataset']['embed_path'],
                                              batch_size=config['dataset']['batch_size'],
                                              rt_load=config['dataset']['rt_load'],
                                              workers=config['dataset']['workers'],
                                              shuffle=True, mode='train',
                                              seed=config['dataset']['seed'])

    val_loader, val_dataset = create_dataloader(val_samples,
                                                config['dataset']['embed_path'],
                                                batch_size=config['dataset']['batch_size'],
                                                rt_load=config['dataset']['rt_load'],
                                                workers=config['dataset']['workers'],
                                                shuffle=False, mode='val',
                                                seed=config['dataset']['seed'])

    # build model architecture, then print to console
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = config.init_obj('loss', module_loss)
    criterion = criterion.to(device)

    writer = WandbWriter(config)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, None, optimizer, 
                      writer=writer,
                      config=config,
                      device=device,
                      data_loader=train_loader,
                      valid_data_loader=val_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
