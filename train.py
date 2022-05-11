import os
import numpy as np
from typing import Dict
from torchvision import utils
from metrics.metric import get_metric
from utils import get_loader, get_model, get_optimizer, get_loss_fn, Timer, Logger
from evaluate import Evaluator
from metrics.metric_tool import ConfuseMatrixMeter
import torch
import wandb

class Trainer(object):
    def __init__(self, config, table):
        self.val_table = table
        self.config = config
        self.train_loader = get_loader(self.config, type='train')
        self.val_loader = get_loader(self.config, type='val')
        self.device = torch.device(f"cuda:{self.config['device']}" if torch.cuda.is_available() else "cpu")
        self.model = get_model(self.config).to(self.device)
        self.optimizer = get_optimizer(self.model, self.config)
        self.loss_fn = get_loss_fn(self.config)
        self.running_metric = ConfuseMatrixMeter(n_class=2)
        self.timer = Timer()
        self.checkpoint_dir = self.config['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger_path = os.path.join(self.checkpoint_dir,'log.txt')
        self.logger = Logger(logger_path)

        self.global_step = 0
        self.steps_per_epoch = len(self.train_loader)
        self.total_steps = self.config['epochs'] * self.steps_per_epoch

        self.epoch = 0
        self.batch_id = 0
        self.best_val_acc = 0.0
        self.epoch_acc = 0.0
        self.best_epoch_id = 0
        self.train_epoch_loss = 0.0
        self.val_epoch_loss = 0.0
        self.is_training = False
        self.batch = None
        self.pred = None
        self.imgA = None
        self.imgB = None
        self.pred = None
        self.label = None


    def _timer_update(self):
        self.global_step = (self.epoch-1) * self.steps_per_epoch + self.batch_id
        self.timer.update_progress((self.global_step + 1) / self.total_steps)
        est = self.timer.estimated_remaining()
        imps = (self.global_step + 1) * self.config['batch_size'] / self.timer.get_stage_elapsed()
        return imps, est

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
        m = len(self.train_loader)
        if self.is_training is False:
            m = len(self.val_loader)
        imps, est = self._timer_update()
        if np.mod(self.batch_id, self.config['train_metric_frequency']) == 1:
            message = 'Is_training: %s. [%d,%d][%d,%d], imps: %.2f, est: %.2fh, training_loss: %.5f, running_mf1: %.5f\n' % \
                      (self.is_training, self.epoch, self.config['epochs'], self.batch_id, m,
                       imps * self.config['batch_size'], est,
                       self.trianing_loss.item(), running_acc)
            self.logger.write(message)
        if self.is_training is True:
            if np.mod(self.batch_id, self.config['val_visual_frequency']) == 1:
                self.val_table.add_data(wandb.Image(self._make_numpy_grid(self.imgA[0,:,:,:])),
                                   wandb.Image(self._make_numpy_grid(self.imgB[0,:,:,:])),
                                   wandb.Image(self._make_numpy_grid(self._visualize_pred())),
                                   wandb.Image(self._make_numpy_grid(self.label[0,:,:])))

    def _collect_epoch_states(self, type):
        scores = self.running_metric.get_scores()
        self.epoch_acc = scores['mf1']
        self.logger.write('Is_training: %s. Epoch %d / %d, epoch_mF1= %.5f\n' %
              (self.is_training, self.epoch, self.config['epochs'], self.epoch_acc))
        if type == 'train':
            message = ''
            for k, v in scores.items():
                message += '%s: %.5f ' % (k, v)
                wandb_ms = {f'train/{k}': v}
                wandb.log(wandb_ms)
            self.logger.write(message+'\n')
            self.logger.write('\n')
            wandb.log({'train/train_loss': self.train_epoch_loss})
        else:
            message = ''
            for k, v in scores.items():
                message += '%s: %.5f ' % (k, v)
                wandb_ms = {f'val/{k}': v}
                wandb.log(wandb_ms)
            self.logger.write(message+'\n')
            self.logger.write('\n')
            wandb.log({'val/val_loss': self.val_epoch_loss})


    def _clear_cache(self):
        self.running_metric.clear()

    def _save_checkpoint(self, ckpt_name):
        torch.save({
            'epoch_id': self.epoch,
            'best_val_acc': self.best_val_acc,
            'best_epoch_id': self.best_epoch_id,
            'model_G_state_dict': self.model.state_dict(),
            'optimizer_G_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.checkpoint_dir, ckpt_name))

    def _update_checkpoints(self):
        # save current model
        self._save_checkpoint(ckpt_name='last_ckpt.pt')
        self.logger.write('Lastest model updated. Epoch_acc=%.4f, Historical_best_acc=%.4f (at epoch %d)\n'
              % (self.epoch_acc, self.best_val_acc, self.best_epoch_id))
        self.logger.write('\n')
        # update the best model (based on eval acc)
        if self.epoch_acc > self.best_val_acc:
            self.best_val_acc = self.epoch_acc
            self.best_epoch_id = self.epoch
            self._save_checkpoint(ckpt_name='best_ckpt.pt')
            self.logger.write('*' * 10 + 'Best model updated!\n')
            self.logger.write('\n')

    def train(self) -> None:

        for self.epoch in range(1, self.config['epochs']+1):
            self.train_epoch_loss = 0.0
            self.val_epoch_loss = 0.0
            self._clear_cache()
            self.is_training =True
            self.model.train()
            for self.batch_id, self.batch in enumerate(self.train_loader, 0):
                self.imgA, self.imgB, self.label = self.batch
                self.imgA = self.imgA.to(self.device, dtype=torch.float32)
                self.imgB = self.imgB.to(self.device, dtype=torch.float32)
                self.label = self.label.to(self.device, dtype=torch.long)
                self.optimizer.zero_grad()
                self.pred = self.model(self.imgA, self.imgB)
                self.trianing_loss = self.loss_fn(self.pred, self.label)
                self.train_epoch_loss += self.trianing_loss.item()
                self.trianing_loss.backward()
                self.optimizer.step()
                self._collect_running_batch_states()
            self._collect_epoch_states('train')


            self.model.eval()
            self.is_training = False
            self._clear_cache()

            for self.batch_id, self.batch in enumerate(self.val_loader, 0):
                self.imgA, self.imgB, self.label = self.batch
                self.imgA = self.imgA.to(self.device, dtype=torch.float32)
                self.imgB = self.imgB.to(self.device, dtype=torch.float32)
                self.label = self.label.to(self.device, dtype=torch.long)
                self.pred = self.model(self.imgA, self.imgB)
                self.val_loss = self.loss_fn(self.pred, self.label)
                self.val_epoch_loss += self.val_loss.item()
                self._collect_running_batch_states()
            self._collect_epoch_states('val')
            self._update_checkpoints()

            wandb.log({'val_table': val_table})



if __name__ == '__main__':

    wandb.init(project='Lapsrn_cd',
               config='config-defaults.yaml')
    wandb.run.name = wandb.config['name']
    val_table = wandb.Table(columns=['imgA', 'imgB', 'pred', 'gt'])
    test_table = wandb.Table(columns=['imgA', 'imgB', 'pred', 'gt'])
    config = wandb.config
    Trainer = Trainer(config, val_table)
    Trainer.train()
    Evaluator = Evaluator(config, test_table)
    Evaluator.evaluate()
    wandb.log({'test_tabel': test_table})
    wandb.finish()