import logging
import os
import torch
import wandb
import numpy as np
from torchvision import utils

from metrics.metric_tool import ConfuseMatrixMeter
from utils import get_loader, get_model, Logger
from metrics.metric import get_metric


class Evaluator():
    def __init__(self, config, table):
        self.test_table = table
        self.config = config
        self.test_loader = get_loader(config, type='test')
        self.device = torch.device(f"cuda:{self.config['device']}" if torch.cuda.is_available() else "cpu")
        self.model = get_model(config)
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        # define logger file
        self.checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger_path = os.path.join(self.checkpoint_dir, 'log_test.txt')
        self.logger = Logger(logger_path)
        self.G_pred = None
        self.pred_vis = None
        self.batch = None
        self.is_training = False
        self.epoch = 0
        self.batch_id = 0
        self.epoch_id = 0


    def _load_checkpoint(self, checkpoint_name='best_ckpt.pt'):

        if os.path.exists(os.path.join(self.checkpoint_dir, checkpoint_name)):
            self.logger.write('loading last checkpoint...\n')
            # load the entire checkpoint
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, checkpoint_name), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_G_state_dict'])
            self.model.to(self.device)

            # update some other states
            self.best_val_acc = checkpoint['best_val_acc']
            self.best_epoch_id = checkpoint['best_epoch_id']

            self.logger.write('Eval Historical_best_acc = %.4f (at epoch %d)\n' %
                  (self.best_val_acc, self.best_epoch_id))
            self.logger.write('\n')
        else:
            raise FileNotFoundError('no such checkpoint %s' % checkpoint_name)

    def _clear_cache(self):
        self.running_metric.clear()

    def _update_metric(self):
        """
        update metric
        """
        target = self.label.to(self.device).detach()
        G_pred = self.pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        current_score = self.running_metric.update_cm(pr=G_pred.cpu().numpy(), gt=target.cpu().numpy())
        return current_score

    def _visualize_pred(self):
        pred = torch.argmax(self.pred[0,:,:,:], dim=0, keepdim=True)
        pred_vis = pred * 255
        return pred_vis

    def _make_numpy_grid(self, tensor_data, pad_value=0, padding=0):
        tensor_data = tensor_data.detach()
        vis = utils.make_grid(tensor_data, pad_value=pad_value, padding=padding)
        vis = np.array(vis.cpu()).transpose((1, 2, 0))
        if vis.shape[2] == 1:
            vis = np.stack([vis, vis, vis], axis=-1)
        return vis

    def _collect_running_batch_states(self):
        running_acc = self._update_metric()
        m = len(self.test_loader)
        if np.mod(self.batch_id, self.config['train_metric_frequency']) == 1:
            message = 'Is_training: %s. running_mf1: %.5f\n' % \
                      (self.is_training, running_acc)
            self.logger.write(message)

        if np.mod(self.batch_id, self.config['test_visual_frequency']) == 1:
            self.test_table.add_data(wandb.Image(self._make_numpy_grid(self.imgA[0, :, :, :])),
                               wandb.Image(self._make_numpy_grid(self.imgB[0, :, :, :])),
                               wandb.Image(self._make_numpy_grid(self._visualize_pred())),
                               wandb.Image(self._make_numpy_grid(self.label[0, :, :])))


    def _collect_epoch_states(self):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
                          (self.is_training, self.epoch, self.config['epochs'], self.epoch_acc))
        message = ''
        for k, v in scores.items():
            message += '%s: %.5f ' % (k, v)
            wandb_ms = {f'test/{k}': v}
            wandb.log(wandb_ms)
        self.logger.write(message + '\n')
        self.logger.write('\n')


    def evaluate(self, checkpoint_name='best_ckpt.pt'):
        self._load_checkpoint(checkpoint_name)

        ################## Eval ##################
        ##########################################
        self.logger.write('Begin evaluation...\n')
        self._clear_cache()
        self.is_training = False
        self.model.eval()

        # Iterate over data.
        for self.batch_id, self.batch in enumerate(self.test_loader, 0):
            with torch.no_grad():
                self.imgA, self.imgB, self.label = self.batch
                self.imgA = self.imgA.to(self.device, dtype=torch.float32)
                self.imgB = self.imgB.to(self.device, dtype=torch.float32)
                self.label = self.label.to(self.device, dtype=torch.long)
                self.pred = self.model(self.imgA, self.imgB)
            self._collect_running_batch_states()
        self._collect_epoch_states()



