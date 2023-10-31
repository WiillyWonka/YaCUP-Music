import argparse
import torch
from tqdm import tqdm
from model.model import BaselineModel
import numpy as np
import pandas as pd
from data_loader.data_loaders import create_dataloader
from utils import prepare_device, load_labels
from torchmetrics.classification import MultilabelAveragePrecision
import seaborn as sns
import matplotlib.pyplot as plt

def predict(model, loader, device):
    model.eval()
    track_idxs = []
    predictions = []
    with torch.no_grad():
        for data in loader:
            track_idx, embeds = data
            embeds = [x.to(device) for x in embeds]
            pred_logits = model(embeds)
            pred_probs = torch.sigmoid(pred_logits)
            predictions.append(pred_probs.cpu().numpy())
            track_idxs.append(track_idx.numpy())
    predictions = np.vstack(predictions)
    track_idxs = np.vstack(track_idxs).ravel()
    return track_idxs, predictions


def main(config):
    config = {
        'n_gpu': 1,
        'checkpoint_path': 'saved/models/BaseLine/1030_175106/best.pth',
        'dataset':
            {
            'embed_path': 'dataset/embeddings/',
            'labels_path': 'dataset/autosplit_train.csv',
            'rt_load': True,
            'batch_size': 64,
            'workers': 8,
            'seed': 0
            }
    }



    samples = load_labels(config['dataset']['labels_path'])
    # Trainloader
    dataloader, dataset = create_dataloader(samples,
                                              config['dataset']['embed_path'],
                                              batch_size=config['dataset']['batch_size'],
                                              rt_load=config['dataset']['rt_load'],
                                              workers=config['dataset']['workers'],
                                              shuffle=True, mode='val',
                                              seed=config['dataset']['seed'])
    
    model = BaselineModel()

    checkpoint = torch.load(config['checkpoint_path'])
    model.load_state_dict(checkpoint['state_dict'])

    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)

    ap_metric = MultilabelAveragePrecision(num_labels=256, average='none').to(device)

    model.eval()
    track_idxs = []
    predictions = []
    with torch.no_grad():
        for data in tqdm(dataloader):
            track_idxs, embeds, target = data

            target = target.to(device)
            embeds = [x.to(device) for x in embeds]
            pred_logits = model(embeds)
            pred_probs = torch.sigmoid(pred_logits)

            ap_metric.update(pred_probs, target.int())
            
    ap_values = ap_metric.compute()

    df = pd.DataFrame.from_dict({i: ap_value.item() for i, ap_value in enumerate(ap_values)}, orient='index', columns=['value'])
    print(df)
    sns.barplot(x=df.index, y='value', data=df)
    plt.show()



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    # config = ConfigParser.from_args(args)
    config=None
    main(config)
