import argparse
import torch
from tqdm import tqdm
from model.model import BaselineModel
import numpy as np
import pandas as pd
from data_loader.data_loaders import create_dataloader
from utils import prepare_device, load_labels


def predict(model, loader, device):
    model.eval()
    track_idxs = []
    predictions = []
    with torch.no_grad():
        for data in tqdm(loader, total=len(loader)):
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
        'checkpoint_path': 'saved/models/BaseLine/1031_152037/best.pth',
        'dataset':
            {
            'embed_path': 'dataset/embeddings/',
            'labels_path': 'dataset/test.csv',
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
                                              shuffle=False, mode='test',
                                              seed=config['dataset']['seed'])
    
    model = BaselineModel()

    checkpoint = torch.load(config['checkpoint_path'])
    model.load_state_dict(checkpoint['state_dict'])

    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)

    track_idxs, predictions = predict(model, dataloader, device)

    predictions_df = pd.DataFrame([
        {'track': track, 'prediction': ','.join([str(p) for p in probs])}
        for track, probs in zip(track_idxs, predictions)
    ])

    predictions_df.to_csv('prediction.csv', index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')

    # config = ConfigParser.from_args(args)
    config=None
    main(config)
