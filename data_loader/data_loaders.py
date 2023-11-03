from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os
from utils.util import seed_worker
import random

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
    def __init__(self, df, embed_path, rt_load=True, augment_enable=False, mode: str = 'train'):
        self.df = df

        self.mode = mode
        self.testing = True if mode == 'test' else False
        self.valid = True if mode == 'val' else False

        self.augment_enable = augment_enable and mode == 'train'

        self.cat_p = 0
        self.n_cat = 4 # must be >= 1
        self.cat_enable = self.augment_enable and self.cat_p > 0

        self.delete_p = 0
        self.n_deletes = 4 # must be >= 1
        self.delete_enable = self.augment_enable and self.delete_p > 0      

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

    def _get_sample(self, idx):
        row = self.df.iloc[idx]
        track_idx = row.track

        if self.rt_load:
            embeds = np.load(self.path_template.format(track_idx=track_idx))
        else:
            embeds = self.embeddings[track_idx]

        if self.testing:
            return track_idx, embeds
        
        tags = [int(x) for x in row.tags.split(',')]
        target = np.zeros(self.NUM_TAGS)
        target[tags] = 1

        return track_idx, embeds, target


    def __getitem__(self, idx):
        if self.testing:
            return self._get_sample(idx)
        
        track_idx, embeds, target = self._get_sample(idx)

        if self.cat_enable and random.random() < self.cat_p:
            track_idx, embeds, target = [track_idx], [embeds], [target]
            for i in range(random.randint(1, self.n_cat)):
                add_track_idx, add_embeds, add_target = self._get_sample(i)
                track_idx.append(add_track_idx)
                embeds.append(add_embeds)
                target.append(add_target)
            
            embeds = np.concatenate(embeds, axis=0)

            result_target = np.zeros_like(target[0])
            for target_item in target:
                result_target = np.logical_or(result_target, target_item)
            target = result_target.astype(float)

        if self.delete_enable and random.random() < self.delete_p:
            random_indexes = np.random.choice(embeds.shape[0], size=random.randint(1, self.n_deletes), replace=False)
            np.delete(embeds, random_indexes, axis=0).shape
                

        return track_idx, embeds, target
    
    @staticmethod
    def collate_fn(b):
        track_idxs = [x[0] for x in b]

        embeds = [torch.from_numpy(x[1]) for x in b]
        
        targets = np.vstack([x[2] for x in b])
        targets = torch.from_numpy(targets)
        return track_idxs, embeds, targets
    
    @staticmethod
    def collate_fn_test(b):
        track_idxs = torch.from_numpy(np.vstack([x[0] for x in b]))
        embeds = [torch.from_numpy(x[1]) for x in b]
        return track_idxs, embeds
