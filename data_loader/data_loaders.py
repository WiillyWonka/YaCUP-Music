from torch.utils.data import Dataset
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from typing import Optional
import logging
from sklearn.model_selection import train_test_split
from base.base_data_module import BaseDataModule


class DataModule(BaseDataModule):
    def __init__(self,
                 batch_size,
                 workers,
                 labels_path,
                 train_size=None,
                 test_size=None,
                 save_split=None,
                 shuffle=False,
                 seed=42,
                 dataset_args = dict()) -> None:
        
        self.train_size = train_size
        self.test_size = test_size
        self.save_split = save_split
        self.labels_path = labels_path

        self.dataset_args = dataset_args

        super().__init__(batch_size,
                         workers,
                         shuffle,
                         seed)

    def setup(self, stage: Optional[str] = None) -> None:

        samples = self._load_labels()
        # save_split = None
        train_samples, val_samples = self._autosplit(samples)

        self.data_train = TaggingDataset(train_samples, mode='train', **self.dataset_args)
        self.data_val = TaggingDataset(val_samples, mode='val', **self.dataset_args)

    def _load_labels(self):
        return pd.read_csv(self.labels_path)
    
    def _autosplit(self, samples):
        random.seed(self.seed)  # for reproducibility
        train_samples, test_samples = train_test_split(samples, 
                                                       train_size=self.train_size, 
                                                       test_size=self.test_size,
                                                       random_state = self.seed )

        if isinstance(self.save_split, (str, Path)):
            save_split = Path(self.save_split)

            for x, save_samples in zip(['autosplit_train.csv', 'autosplit_val.csv'], [train_samples, test_samples]):
                file_path = save_split / x 
                
                if file_path.exists():
                    logging.warning(f"Older splitting file {str(file_path)} exists! It will be replaced by newer file.")
                    file_path.unlink()  # remove existing

                save_samples.to_csv(file_path, index=False)

        return train_samples, test_samples


class TaggingDataset(Dataset):
    NUM_TAGS = 256
    def __init__(self, 
                 df, 
                 embed_path, 
                 rt_load=True, 
                 augment_enable=False, 
                 mode: str = 'train',
                 cat_p=0,
                 n_cat=0,
                 delete_p=0,
                 f_deletes=0,
                 dropout_p=0,
                 f_dropout=0,
                 jitter_p=0,
                 f_jitter=0,
                 std_embeddings: str = "embeddings_std.npy",
                 mean_embeddings: str = "embeddings_mean.npy",
                 permutation_p=0,
                 n_permutation=0,
                 normalize=True
                 ):
        
        self.df = df

        self.mode = mode
        self.testing = True if mode == 'test' else False
        self.valid = True if mode == 'val' else False

        self.augment_enable = augment_enable and mode == 'train'

        self.cat_p = cat_p
        self.n_cat = n_cat # must be >= 1
        self.cat_enable = self.augment_enable and self.cat_p > 0

        self.delete_p = delete_p # probability [0, 1]
        self.f_deletes = f_deletes # fraction [0, 1]
        self.delete_enable = self.augment_enable and self.delete_p > 0 and self.f_deletes > 0

        self.dropout_p = dropout_p # probability [0, 1]
        self.f_dropout = f_dropout # fraction [0, 1]
        self.dropout_enable = self.augment_enable and self.dropout_p > 0 and self.f_dropout > 0

        self.jitter_p = jitter_p # probability [0, 1]
        self.f_jitter = f_jitter # fraction [0, 1]
        self.jitter_enable = self.augment_enable and self.jitter_p > 0 and self.f_jitter > 0

        self.permutation_p = permutation_p # probability [0, 1]
        self.n_permutation = n_permutation # pair numbers
        self.permutation_enable = self.augment_enable and self.permutation_p > 0 and self.n_permutation > 0

        self.std_embeddings = np.load(std_embeddings)
        self.mean_embeddings = np.load(mean_embeddings)

        self.normalize = normalize

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

        if self.normalize:
            embeds = (embeds - self.mean_embeddings) / self.std_embeddings

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
            for _ in range(random.randint(1, self.n_cat)):
                i = random.randint(1, len(self)-1)
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
            size = int(embeds.shape[0] * random.random() * self.f_deletes)
            random_indexes = np.random.choice(embeds.shape[0], size=size, replace=False)
            embeds = np.delete(embeds, random_indexes, axis=0)

        if self.permutation_enable and random.random() < self.permutation_p:
            size = int(random.random() * self.n_permutation)
            for _ in range(size):
                i = np.random.choice(range(embeds.shape[0]))
                j = np.random.choice(range(embeds.shape[0]))

                embeds[i], embeds[j] = embeds[j], embeds[i]

        if self.dropout_enable and random.random() < self.dropout_p:
            size = int(embeds.size * random.random() * self.f_dropout)

            idxs = np.random.choice(range(embeds.size), size=size, replace=False)
            for idx in idxs:
                i, j = idx // embeds.shape[1], idx % embeds.shape[1]
                embeds[i, j] = 0
                
        if self.jitter_enable and random.random() < self.jitter_p:
            for i in range(embeds.shape[0]):
                fraction_sign = np.sign(random.random() - 0.5)
                fraction_magnitude = random.random() * self.f_jitter
                if not self.normalize:
                    fraction_magnitude *= self.std_embeddings
                embeds[i] += fraction_sign * fraction_magnitude

        return track_idx, embeds, target
    
    @staticmethod
    def collate_fn(b):
        track_idxs = [x[0] for x in b]

        embeds = [torch.from_numpy(x[1]) for x in b]
        
        if len(b[0]) > 2:
            targets = np.vstack([x[2] for x in b])
            targets = torch.from_numpy(targets)

            return track_idxs, embeds, targets
        
        return track_idxs, embeds
