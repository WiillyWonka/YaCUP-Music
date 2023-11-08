from abc import ABC, abstractmethod
from typing import Optional, Any
import os
import torch
from torch.utils.data import DataLoader
from utils.util import seed_worker


class BaseDataModule(ABC):
    def __init__(self, 
                 batch_size,
                 workers,
                 shuffle,
                 seed) -> None:
        
        self.batch_size = batch_size
        self.workers = workers
        self.shuffle = shuffle
        self.seed = seed

        self.setup()

    @abstractmethod
    def setup(self,  stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        pass

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """

        return self._create_dataloader(self.data_train)

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """

        return self._create_dataloader(self.data_val)
    
    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return self._create_dataloader(self.data_test)

    def _create_dataloader(self, dataset):

        batch_size = min(self.batch_size, len(dataset))
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, self.workers])  # number of workers
        loader = InfiniteDataLoader
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + self.seed)
        return loader(dataset,
                    batch_size=batch_size,
                    shuffle=self.shuffle,
                    num_workers=nw,
                    pin_memory=True,
                    collate_fn=dataset.collate_fn,
                    worker_init_fn=seed_worker,
                    generator=generator)

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
