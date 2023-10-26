import wandb
import os


class WandbWriter:
    def __init__(self, config) -> None:
        with open("wandb_api_key", 'r') as f:
            key = f.read().strip('\n')
            os.environ["WANDB_API_KEY"] = key

        self.run = wandb.init(
            config=config, 
            resume="allow",
            project=config['project'],
            name=config['name'],
            settings=wandb.Settings(start_method="fork")
        )

        self.run.define_metric("epoch")

        self.log_data = {}
        self.epoch = 0

    def log_metric(self, key, value, epoch=None):
        self.log_data[key] = value
        self.epoch = epoch

    def flush(self, epoch=None):
        self.run.log(self.log_data, step=self.epoch if epoch is None else epoch)

        