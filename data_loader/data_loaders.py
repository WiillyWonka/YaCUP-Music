from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
from utils.util import seed_worker

def create_dataloader(df,
                      embed_path,
                      batch_size,
                      rt_load=True,
                      mode='train',
                      workers=8,
                      shuffle=False,
                      seed=0):
    
    dataset = TaggingDataset(df, embed_path, rt_load, mode)

    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])  # number of workers
    loader = InfiniteDataLoader
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed)
    return loader(dataset,
                  batch_size=batch_size,
                  shuffle=shuffle,
                  num_workers=nw,
                  pin_memory=True,
                  collate_fn=TaggingDataset.collate_fn_test if mode == 'test' else TaggingDataset.collate_fn,
                  worker_init_fn=seed_worker,
                  generator=generator), dataset


class InfiniteDataLoader(DataLoader):
    """ Dataloader that reuses workers

    Uses same syntax as vanilla DataLoader
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler:
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class TaggingDataset(Dataset):
    NUM_TAGS = 256
    def __init__(self, df, embed_path, rt_load=True, mode: str = 'train'):
        self.df = df

        self.mode = mode
        self.testing = True if mode == 'test' else False
        self.valid = True if mode == 'val' else False

        self.rt_load = rt_load
        if rt_load:
            self.path_template = embed_path + '/{track_idx}.npy'
        else:
            self.embeddings = {}
            for embed_file in tqdm(list(Path("embeddings").iterdir())):
                embed_idx = int(embed_file.stem)
                self.embeddings[embed_idx] = np.load(embed_file)
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        track_idx = row.track

        if self.rt_load:
            embeds = np.load(self.path_template.format(track_idx=track_idx))
        else:
            embeds = self.embeddings[idx]

        embeds /= 4

        if self.testing:
            return track_idx, embeds
        
        tags = [int(x) for x in row.tags.split(',')]
        target = np.zeros(self.NUM_TAGS)
        target[tags] = 1
        return track_idx, embeds, target
    
    @staticmethod
    def collate_fn(b):
        track_idxs = torch.from_numpy(np.vstack([x[0] for x in b]))

        max_seq = max([x[1].shape[0] for x in b])
        padded_embeds = [np.pad(x[1], ((0, max_seq - x[1].shape[0]), (0, 0)), 'constant') for x in b]
        embeds = np.stack(padded_embeds)
        embeds = torch.from_numpy(embeds)

        # embeds = [torch.from_numpy(x[1]) for x in b]
        
        targets = np.vstack([x[2] for x in b])
        targets = torch.from_numpy(targets)
        return track_idxs, embeds, targets
    
    @staticmethod
    def collate_fn_test(b):
        track_idxs = torch.from_numpy(np.vstack([x[0] for x in b]))
        embeds = [torch.from_numpy(x[1]) for x in b]
        return track_idxs, embeds
