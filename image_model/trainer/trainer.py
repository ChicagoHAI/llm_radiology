import numpy as np
import torch
from torchvision.utils import make_grid
# from base import BaseTrainer
# from utils import inf_loop, MetricTracker
import torch
from abc import abstractmethod
from numpy import inf
# from logger import TensorboardWriter
import logging
import json
import torch
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from tqdm import tqdm
import os
os.environ["WANDB_DIR"] = "/net/scratch/chacha/future_of_work/report-generation/wandb"
os.environ["WANDB_CACHE_DIR"] = "/net/scratch/chacha/future_of_work/report-generation/wandb"
os.environ["WANDB_CONFIG_DIR"] = "/net/scratch/chacha/future_of_work/report-generation/wandb"

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, logger, model, criterion, metric_ftns, optimizer, config, wandb, save_dir, lr_scheduler):
        self.args = config
        # self.logger = config.get_logger('trainer', config['trainer']['verbosity'])
        self.logger = logger
        self.model = model
        if not self.args.training.single_gpu:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        
        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer
        self.wb_run = wandb
        self.wb_run.watch(model, criterion, log='all', log_freq=100)

        cfg_trainer = self.args.training #config['trainer']
        self.epochs = cfg_trainer.epochs #['epochs']
        self.save_period = cfg_trainer.save_period # ['save_period']
        self.monitor = cfg_trainer.monitor
        self.lr_scheduler = lr_scheduler
        # 'off' #cfg_trainer.get('monitor', 'off')

        # configuration to monitor model performance and save best
        if not self.monitor:
            self.mnt_mode = 'off'
            self.mnt_best = 0
        else:
            # self.mnt_mode, self.mnt_metric = self.monitor.split()
            self.mnt_mode = 'max'
            self.mnt_metric = 'val_multi_label_accuracy'
            assert self.mnt_mode in ['min', 'max']

            self.mnt_best = inf if self.mnt_mode == 'min' else -inf
            self.early_stop = cfg_trainer.patience
            # get('early_stop', inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1

        # self.checkpoint_dir = Path(config.training.save_dir)
        # post_fix = '_debug' if self.args.training.debug else ''
        self.checkpoint_dir = Path(save_dir)
        ensure_dir(self.checkpoint_dir) # create checkpoint dir if not exist

        # setup visualization writer instance                
        # self.writer = TensorboardWriter(config.log_dir, self.logger, cfg_trainer['tensorboard'])

        if self.args.training.resume_from_checkpoint:
            ## find model file from saved dir 
            ## filter out all files that do not have .pth extension
            checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pth') and f.startswith('checkpoint')]
            ## sort based on epoch number and get the largest file
            if len(checkpoint_files) > 0:
                checkpoint_files.sort(key=lambda x: int(x.split('epoch')[1].split('.')[0]))
                checkpoint_file = checkpoint_files[-1]
                print(f"Resuming from checkpoint file: {checkpoint_file}")
                self.logger.info(f"Resuming from checkpoint file: {checkpoint_file}")
                self._resume_checkpoint(os.path.join(self.checkpoint_dir, checkpoint_file))

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def train(self):
        """
        Full training logic
        """
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            result = self._train_epoch(epoch)
            # self.wb_run.log(result)

            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)
            self.wb_run.log(log)
            # self.wb_run.log(result)

            # print logged informations to the screen
            if epoch % 10 == 0:
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            # evaluate model performance according to configured metric, save best checkpoint as model_best
            best = False
            if self.mnt_mode != 'off':
                try:
                    # check whether model performance improved or not, according to specified metric(mnt_metric)
                    improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                               (self.mnt_mode == 'max' and log[self.mnt_metric] > self.mnt_best)
                except KeyError:
                    self.logger.warning("Warning: Metric '{}' is not found. "
                                        "Model performance monitoring is disabled.".format(self.mnt_metric))
                    self.mnt_mode = 'off'
                    improved = False

                if improved:
                    self.mnt_best = log[self.mnt_metric]
                    not_improved_count = 0
                    best = True
                else:
                    not_improved_count += 1

                if not_improved_count > self.early_stop:
                    self.logger.info("Validation performance didn\'t improve for {} epochs. "
                                     "Training stops.".format(self.early_stop))
                    break

            # if epoch % self.save_period == 0:
            if self.device == 0 and epoch % self.save_period == 0:
                self._save_checkpoint(epoch, save_best=best)
            
            if self.device == 0 and best:
                self._save_checkpoint(epoch, save_best=best)

    def _save_checkpoint(self, epoch, save_best=False):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.mnt_best,
            'config': self.config,
            'scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))
        if save_best: ## TODO save best
            best_path = str(self.checkpoint_dir / 'model_best.pth')
            torch.save(state, best_path)
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config'].model.arch != self.args.model.arch:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        ## TODO when we have different optimizers
        # if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
        #     self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
        #                         "Optimizer parameters not being resumed.")
        # else:
        self.logger.info("Resuming optimizer from checkpoint")
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.logger.info("Resuming scheduler from checkpoint")
        self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
        

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, logger, model, criterion, metric_ftns, optimizer, config, wandb, save_dir, device,
                 data_loader, valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(logger, model, criterion, metric_ftns, optimizer, config, wandb, save_dir, lr_scheduler)
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
        # self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
                                        #    , writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns])
                                        #    , writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        with tqdm(total=len(self.data_loader.dataset)) as pbar:
            tqdm_interval = 0
            pbar.set_description('Epoch {0}/{1}'.format(epoch, self.args.training.epochs))
            if not self.args.training.single_gpu:
                self.data_loader.sampler.set_epoch(epoch)
            for batch_idx, (ids, img, txt, label, vp) in enumerate(self.data_loader):
                
                tqdm_interval += img.shape[0]
                pbar.update(tqdm_interval)
                # for batch_idx, (img, txt, target) in enumerate(self.data_loader):
                # continue
                # img, txt, label = img.to(self.device), txt.to(self.device), label.to(self.device)
                # TODO txt
                img, label = img.to(self.device), label.to(self.device)
                
                # data = img, txt
                # data, label = data.to(self.device), label.to(self.device)

                self.optimizer.zero_grad()
                output, _ = self.model(img) # TODO txt
                loss = self.criterion(output, label)
                loss.backward()
                self.optimizer.step()

                # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.train_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(output, label))

                if batch_idx % self.log_step == 0:
                    self.logger.debug('Train Epoch: {} {} Loss: {:.6f} accuracy: {:.6f}'.format(
                        epoch,
                        self._progress(batch_idx),
                        loss.item(),
                        self.train_metrics.result()['multi_label_accuracy']))
                    # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))

                # if batch_idx == self.len_epoch:
                #     break
        
        log = self.train_metrics.result()

        if self.do_validation & (epoch % self.args.training.val_freq == 0):
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            # for batch_idx, (data, target) in enumerate(self.valid_data_loader):
            if not self.args.training.single_gpu:
                self.valid_data_loader.sampler.set_epoch(epoch)

            for batch_idx, (ids, img, txt, label, vp) in enumerate(self.valid_data_loader):
                # img, txt, target = img.to(self.device), txt.to(self.device), target.to(self.device)
                
                img, label = img.to(self.device), label.to(self.device)
   
                # data, target = data.to(self.device), target.to(self.device)

                output, _ = self.model(img)
                loss = self.criterion(output, label)

                # self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, label))
                # self.writer.add_image('input', make_grid(data.cpu(), nrow=8, normalize=True))
        # print(self.valid_metrics.result())
        ## why there is randomness in the validation loss?
        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)