import numpy as np
import torch
from base import BaseTrainer
from utils import inf_loop
import torchmetrics as tm
from tqdm import tqdm


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    NUM_TAGS = 256

    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, writer, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, writer, config)
        self.config = config
        self.device = device
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.loss_metric = tm.MeanMetric(nan_strategy="error").to(device)
        self.running_loss_metric = tm.aggregation.RunningMean(window = 10, nan_strategy="error").to(device)
        self.ap_metric = tm.classification.MultilabelAveragePrecision(num_labels=self.NUM_TAGS, average='macro').to(device)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.loss_metric.reset()
        self.running_loss_metric.reset()

        pbar = tqdm(enumerate(self.data_loader), total=self.len_epoch)
        for batch_idx, (data) in pbar:
            track_idxs, embeds, target = data
            # embeds = [x.to(self.device) for x in embeds]
            embeds = embeds.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            pred_logits = self.model(embeds)
            pred_probs = torch.sigmoid(pred_logits)

            loss = self.criterion(pred_logits, target)
            loss.backward()
            self.optimizer.step()

            self.loss_metric.update(loss.item())
            self.running_loss_metric.update(loss.item())

            if batch_idx == self.len_epoch:
                break

            pbar.set_description("train/loss: {loss:3f}".format(loss=self.running_loss_metric.compute()))
            
        log_dict = {"train/loss": self.loss_metric.compute().item()}

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log_dict.update(val_log)

        self.writer.log(log_dict)
        self.writer.flush(epoch)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log_dict

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()

        self.loss_metric.reset()
        self.ap_metric.reset()
        self.running_loss_metric.reset()

        with torch.no_grad():
            pbar = tqdm(enumerate(self.valid_data_loader), total=len(self.valid_data_loader))
            for batch_idx, (data) in pbar:
                track_idxs, embeds, target = data
                # embeds = [x.to(self.device) for x in embeds]
                embeds = embeds.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                pred_logits = self.model(embeds)
                pred_probs = torch.sigmoid(pred_logits)

                loss = self.criterion(pred_logits, target)

                self.loss_metric.update(loss.item())
                self.running_loss_metric.update(loss.item())
                self.ap_metric.update(pred_probs, target.int())

                pbar.set_description("val/loss: {loss:3f}".format(loss=self.running_loss_metric.compute()))

        val_metrics = {
            "val/loss": self.loss_metric.compute().item(),
            "val/AP": self.ap_metric.compute().item()
        }

        return val_metrics
