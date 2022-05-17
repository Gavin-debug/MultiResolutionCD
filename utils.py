import sys
import time

import torch
import torch.nn as nn
from typing import Dict
from datasets.dataset import CDDataset
from torch.utils.data import DataLoader
# from models.multi_conv import Multi_conv
from models.BIT import define_G, BASE_Transformer
from models.FPT import FPT
from models.Lapsrn import Lapsrn
from models.multi_conv import Multi_conv
from losses.crossentropy import cross_entropy

def get_loader(config: Dict, type: str) -> DataLoader:
    if type == 'train':
        dataset = CDDataset(config['train_dir'])
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    elif type == 'val':
        dataset = CDDataset(config['val_dir'])
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    elif type == 'test':
        dataset = CDDataset(config['test_dir'])
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    else:
        raise TypeError(f"{type} is invalid, shoule be 'train', 'val' or 'test'")

    return dataloader

def get_model(config: Dict) -> nn.Module:
    if config['model'] == 'Lapsrn':
        model = Lapsrn()
    elif config['model'] == 'Multi_conv':
        model = Multi_conv()
    elif config['model'] == 'BIT':
        model = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4,
                             with_pos='learned', enc_depth=1, dec_depth=8)
    elif config['model'] == 'FPT':
        model = FPT()
    else:
        raise NotImplementedError(f"{config['model']} is not implemented")

    return model

def get_optimizer(model: nn.Module, config: Dict):
    if config['optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=5e-4)
    elif config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    else:
        raise  NotImplementedError(f"{config['optimizer']} is not implemented")

    return optimizer

def get_loss_fn(config:Dict):
    if config['loss_fn'] == 'CrossEntropy':
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"{config['loss_fn']} is not implemented")
    return loss_fn


class Timer:
    def __init__(self, starting_msg = None):
        self.start = time.time()
        self.stage_start = self.start

        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)


    def str_estimated_complete(self):
        return str(time.ctime(self.est_finish))

    def str_estimated_remaining(self):
        return str(self.est_remaining/3600) + 'h'

    def estimated_remaining(self):
        return self.est_remaining/3600

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def lapse(self):
        out = time.time() - self.stage_start
        self.stage_start = time.time()
        return out

class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log_path = outfile
        now = time.strftime("%c")
        self.write('================ (%s) ================\n' % now)

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, mode='a') as f:
            f.write(message)

    def write_dict(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %.7f ' % (k, v)
        self.write(message)

    def write_dict_str(self, dict):
        message = ''
        for k, v in dict.items():
            message += '%s: %s ' % (k, v)
        self.write(message)

    def flush(self):
        self.terminal.flush()