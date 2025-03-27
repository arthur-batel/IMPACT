import functools
import logging
import os
import sys
import warnings
from abc import ABC, abstractmethod
import time
from mailbox import Error

from sklearn.metrics import roc_auc_score, accuracy_score

from IMPACT.dataset import *
import torch
import numpy as np
from torch.utils import data
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
import pandas as pd

from sklearn.metrics import r2_score

import tkinter as tk

from IMPACT.utils import utils


class AbstractModel(ABC):
    def __init__(self, name: str = None, **config):
        super().__init__()

        utils.set_seed(config['seed'])

        self.config = config
        self._name = name
        self.model = None
        self.state = None
        self._trained = False
        self.fold = config['i_fold']

        # # Save/Load model params setup
        # if self.config['save_params']:
        #     self._setup_params_paths()
        #
        # if self.config['load_params']:
        #     self._ask_loading_pref()

        # Tensorboard configuration
        # self._ts = lambda train_loss, valid_loss, valid_rmse,valid_mae, ep: None
        if self.config["tensorboard"]:
            self.writer = SummaryWriter("../tensorboard")
            now = datetime.now()
            self.timestamp = now.strftime("%Y-%m-%d_%H-%M")

            if self.config["flush_freq"]:
                self._ts = self._flush_tensorboard_saving
            else:
                self._ts = self._tensorboard_saving

        # Decide on training method
        if self.config['early_stopping']:

            # Decide on the level of verbosity
            if self.config['verbose_early_stopping']:
                # Decide on the early stopping criterion
                match self.config['esc']:
                    case 'objectives':
                        self._train_method = self._verbose_train_early_stopping_objectives
                    case 'delta_error':
                        self._train_method = self._verbose_train_early_stopping_delta_error
                    case 'error':
                        self._train_method = self._verbose_train_early_stopping_error
                    case 'loss':
                        self._train_method = self._verbose_train_early_stopping_loss

                    case _:
                        logging.warning("Loss improvement selected by default as early stopping criterion")
                        self._train_method = self._verbose_train_early_stopping_loss
            else:
                # Decide on the early stopping criterion
                match self.config['esc']:
                    case 'objectives':
                        self._train_method = self._train_early_stopping_objectives
                    case 'delta_error':
                        self._train_method = self._train_early_stopping_delta_error
                    case 'error':
                        self._train_method = self._train_early_stopping_error
                    case 'loss':
                        self._train_method = self._train_early_stopping_loss
                    case _:
                        logging.warning("Loss improvement selected by default as early stopping criterion")
                        self._train_method = self._train_early_stopping_loss

        else:
            self._train_method = self._train_no_early_stopping

        self.pred_metrics = config['pred_metrics'] if config['pred_metrics'] else ['rmse', 'mae']
        self.pred_metric_functions = {
            'rmse': root_mean_squared_error,
            'mae': mean_absolute_error,
            'r2': r2,
            'mi_acc': micro_ave_accuracy,
            'mi_prec': micro_ave_precision,
            'mi_rec': micro_ave_recall,
            'mi_f1': micro_ave_f1,
            'mi_auc': micro_ave_auc,
        }
        assert set(self.pred_metrics).issubset(self.pred_metric_functions.keys())

        self.profile_metrics = config['profile_metrics'] if config['profile_metrics'] else ['pc-er', 'doa']
        self.profile_metric_functions = {
            'pc-er': compute_pc_er,
            'doa': compute_doa,
            'rm': compute_rm,
        }
        assert set(self.profile_metrics).issubset(self.profile_metric_functions.keys())

        match config['valid_metric']:
            case 'rmse':
                self.valid_metric = root_mean_squared_error
                self.metric_sign = 1  # Metric to minimize :1; metric to maximize :-1
            case 'mae':
                self.valid_metric = mean_absolute_error
                self.metric_sign = 1
            case 'mi_acc':
                self.valid_metric = micro_ave_accuracy
                self.metric_sign = -1

    def train(self, train_data: Dataset, valid_data: Dataset):
        """Train the model."""

        if self.state != "model_initialized":
            raise Exception("The model must be initialized before training")

        lr = self.config['learning_rate']
        batch_size = self.config['batch_size']
        device = self.config['device']

        torch.cuda.empty_cache()

        self.model.to(device, non_blocking=True)

        logging.info('train on {}'.format(device))
        logging.info("-- START Training --")

        self.best_epoch = 0
        self.best_valid_loss = 100000
        self.best_valid_metric = self.metric_sign * 100000

        self.best_model_params = self.model.state_dict()

        train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False)
        # Choosing an appropriate batch_size for the validation data
        valid_batch_size = 20000
        if len(valid_data) < valid_batch_size:
            valid_batch_size = len(valid_data)
        elif abs(len(valid_data) - valid_batch_size) < 500:
            valid_batch_size = len(valid_data) // 2 + 1
        valid_loader = data.DataLoader(valid_data, batch_size=valid_batch_size, shuffle=False, pin_memory=False)

        self.U_mean = self.precompute_user_average_resp(valid_data, device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Reduce the learning rate when a metric has stopped improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        scaler = torch.amp.GradScaler('cuda')

        self.model.train()

        # Call the selected training method
        self._train_method(train_loader, valid_loader, valid_data, optimizer, scheduler, scaler)

        self.model.to(self.config['device'])
        self._trained = True

        logging.info("-- END Training --")

        if self.config['save_params']:
            self._save_user_emb()
            self._save_model_params(temporary=False)
            logging.info("Params saved")
        self.fold += 1

    def precompute_user_average_resp(self, dataloader, device):
        U_resp_sum = torch.zeros(size=(self.model.user_n, self.model.concept_n)).to(device, non_blocking=True)
        U_resp_nb = torch.zeros(size=(self.model.user_n, self.model.concept_n)).to(device, non_blocking=True)

        self.model.eval()
        with torch.no_grad(), torch.amp.autocast('cuda'):
            data_loader = data.DataLoader(dataloader, batch_size=1, shuffle=False)
            for data_batch in data_loader:
                user_ids = data_batch[:, 0].long()
                item_ids = data_batch[:, 1].long()
                labels = data_batch[:, 2]
                dim_ids = data_batch[:, 3].long()

                U_resp_sum[user_ids, dim_ids] += labels
                U_resp_nb[user_ids, dim_ids] += torch.ones_like(labels)

            return U_resp_sum / U_resp_nb

    def _esc(self, valid_loader, valid_data, ep, scheduler):  # Early Stopping Criterion
        pass

    def _train_early_stopping_loss(self, train_loader, valid_loader, valid_data, optimizer, scheduler, scaler):
        epochs = self.config['num_epochs']
        eval_freq = self.config['eval_freq']
        patience = self.config['patience']
        device = self.config['device']

        for _, ep in tqdm(enumerate(range(epochs + 1)), total=epochs, disable=self.config['disable_tqdm']):
            for data_batch in train_loader:
                user_ids = data_batch[:, 0].long()
                item_ids = data_batch[:, 1].long()
                labels = data_batch[:, 2]
                dim_ids = data_batch[:, 3].long()

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self._compute_loss(user_ids, item_ids, dim_ids, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Early stopping
            if (ep + 1) % eval_freq == 0:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    valid_loss, valid_metric = self.evaluate_valid(valid_loader, valid_data.log_tensor)

                    # Checking loss improvement
                    if self.best_valid_loss > valid_loss:
                        self.best_epoch = ep
                        self.best_valid_metric = valid_metric
                        self.best_valid_loss = valid_loss
                        self.best_model_params = self.model.state_dict()

                        scheduler.step(valid_loss)

                    if ep - self.best_epoch >= patience:
                        break
        self.model.load_state_dict(self.best_model_params)

    def _verbose_train_early_stopping_loss(self, train_loader, valid_loader, valid_data, optimizer, scheduler, scaler):
        epochs = self.config['num_epochs']
        eval_freq = self.config['eval_freq']
        patience = self.config['patience']
        device = self.config['device']

        for ep in range(epochs):
            loss_list = []
            for _, data_batch in tqdm(enumerate(train_loader), total=len(train_loader),
                                      disable=self.config['disable_tqdm']):
                user_ids, item_ids, dim_ids, labels = data_batch

                user_ids = user_ids.to(device, non_blocking=True)
                item_ids = item_ids.to(device, non_blocking=True)
                dim_ids = dim_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self._compute_loss(user_ids, item_ids, dim_ids, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_list.append(loss.detach().cpu().item())

            # Early stopping
            if (ep + 1) % eval_freq == 0:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    train_loss = np.mean(loss_list)
                    valid_loss, valid_metric = self.evaluate_valid(valid_loader, valid_data.log_tensor)

                    # Checking loss improvement
                    if valid_loss < self.best_valid_loss:
                        self.best_epoch = ep
                        self.best_valid_metric = valid_metric
                        self.best_valid_loss = valid_loss
                        self.best_model_params = self.model.state_dict()

                        scheduler.step(valid_loss)

                    logging.info(
                        'Epoch [{}] \n- Losses : train={:.4f}, valid={:.4f}, best_valid={:.4f} \n- {}   :       -       '
                        'valid={:.4f}'.format(
                            ep, train_loss, valid_loss, self.best_valid_loss, self.config['valid_metric'], valid_metric,
                            self.best_valid_metric,
                            -1, self.best_valid_doa))

                    if ep - self.best_epoch >= patience:
                        break

        self.model.load_state_dict(self.best_model_params)

    def _verbose_train_early_stopping_error(self, train_loader, valid_loader, valid_data, optimizer, scheduler,
                                            scaler):
        raise NotImplementedError

    def _train_early_stopping_objectives(self, train_loader, valid_loader, valid_data, optimizer, scheduler, scaler):
        epochs = self.config['num_epochs']
        eval_freq = self.config['eval_freq']
        patience = self.config['patience']
        device = self.config['device']

        for ep in range(
                epochs):  # _,ep in tqdm(enumerate(range(epochs + 1)), total=epochs, disable=self.config['disable_tqdm']) :
            for data_batch in train_loader:
                user_ids = data_batch[:, 0].long()
                item_ids = data_batch[:, 1].long()
                dim_ids = data_batch[:, 3].long()
                labels = data_batch[:, 2]

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self._compute_loss(user_ids, item_ids, dim_ids, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Early stopping
            if (ep + 1) % eval_freq == 0:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    valid_loss, valid_metric = self.evaluate_valid(valid_loader, valid_data.log_tensor)
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        valid_doa = compute_pc_er(self.model.concept_n, self.U_mean, self.get_user_emb())
                        valid_doa = valid_doa[valid_doa != 0].mean()

                    if (self.metric_sign * self.best_valid_metric > self.metric_sign * valid_metric) or (
                            valid_doa > self.best_valid_doa):
                        self.best_epoch = ep
                        self.best_valid_loss = valid_loss
                        self.best_valid_metric = valid_metric
                        self.best_valid_doa = valid_doa
                        self.best_model_params = self.model.state_dict()

                        scheduler.step(valid_loss)

                    if ep - self.best_epoch >= patience:
                        break
        self.model.load_state_dict(self.best_model_params)

    def _train_early_stopping_delta_error(self, train_loader, valid_loader, valid_data, optimizer, scheduler, scaler):
        epochs = self.config['num_epochs']
        eval_freq = self.config['eval_freq']
        patience = self.config['patience']
        device = self.config['device']

        for _, ep in tqdm(enumerate(range(epochs + 1)), total=epochs, disable=self.config['disable_tqdm']):
            for data_batch in train_loader:
                user_ids = data_batch[:, 0].long()
                item_ids = data_batch[:, 1].long()
                labels = data_batch[:, 2]
                dim_ids = data_batch[:, 3].long()

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self._compute_loss(user_ids, item_ids, dim_ids, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Early stopping
            if (ep + 1) % eval_freq == 0:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    valid_loss, valid_metric = self.evaluate_valid(valid_loader, valid_data.log_tensor)

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")

                    # Checking loss improvement
                    if self.metric_sign * (self.best_valid_metric - valid_metric) / abs(
                            self.best_valid_metric) > 0.001 or (self.best_valid_mae - valid_mae) / abs(
                            self.best_valid_mae) > 0.001:
                        self.best_epoch = ep
                        self.best_valid_metric = valid_metric
                        self.best_model_params = self.model.state_dict()

                        scheduler.step(valid_loss)

                    if ep - self.best_epoch >= patience:
                        break
        self.model.load_state_dict(self.best_model_params)

    def _train_early_stopping_error(self, train_loader, valid_loader, valid_data, optimizer, scheduler,
                                    scaler):
        epochs = self.config['num_epochs']
        eval_freq = self.config['eval_freq']
        patience = self.config['patience']

        for _, ep in tqdm(enumerate(range(epochs + 1)), total=epochs, disable=self.config['disable_tqdm']):
            for data_batch in train_loader:
                user_ids = data_batch[:, 0].long()
                item_ids = data_batch[:, 1].long()
                dim_ids = data_batch[:, 3].long()
                labels = data_batch[:, 2]

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self._compute_loss(user_ids, item_ids, dim_ids, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Early stopping
            if (ep + 1) % eval_freq == 0:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    valid_loss, valid_metric = self.evaluate_valid(valid_loader, valid_data.log_tensor)

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")

                    # Checking loss improvement
                    if self.metric_sign * self.best_valid_metric > self.metric_sign * valid_metric:  # (self.best_valid_metric - valid_rmse) / abs(self.best_valid_metric) > 0.001:
                        self.best_epoch = ep
                        self.best_valid_metric = valid_metric
                        self.best_model_params = self.model.state_dict()

                        scheduler.step(valid_loss)

                    if ep - self.best_epoch >= patience:
                        break
        self.model.load_state_dict(self.best_model_params)

    def _verbose_train_early_stopping_objectives(self, train_loader, valid_loader, valid_data, optimizer, scheduler,
                                                 scaler):
        epochs = self.config['num_epochs']
        eval_freq = self.config['eval_freq']
        patience = self.config['patience']
        device = self.config['device']

        for ep in range(epochs):
            loss_list = []
            for _, data_batch in tqdm(enumerate(train_loader), total=len(train_loader),
                                      disable=self.config['disable_tqdm']):
                user_ids, item_ids, dim_ids, labels = data_batch

                user_ids = user_ids.to(device, non_blocking=True)
                item_ids = item_ids.to(device, non_blocking=True)
                dim_ids = dim_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self._compute_loss(user_ids, item_ids, dim_ids, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                loss_list.append(loss.detach().cpu().item())

            # Early stopping
            if (ep + 1) % eval_freq == 0:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    train_loss = np.mean(loss_list)
                    valid_loss, valid_metric = self.evaluate_valid(valid_loader, valid_data.log_tensor)
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        valid_doa = utils.evaluate_doa(self.get_user_emb().cpu().numpy(), valid_data.log_tensor.numpy(),
                                                       valid_data.metadata, valid_data.concept_map)
                        valid_doa = valid_doa[valid_doa != 0].mean()

                    # Checking loss improvement
                    if self.metric_sign * (self.best_valid_metric - valid_metric) / abs(
                            self.best_valid_metric) > 0.0001 or \
                            (valid_doa - self.best_valid_doa) / abs(self.best_valid_doa) > 0.0001:
                        self.best_epoch = ep
                        self.best_valid_loss = valid_loss
                        self.best_valid_metric = valid_metric
                        self.best_valid_doa = valid_doa
                        self.best_model_params = self.model.state_dict()

                        scheduler.step(valid_loss)

                    logging.info(
                        'Epoch [{}] \n- Losses : train={:.4f}, valid={:.4f}, best_valid={:.4f} \n- {}   :       -       '
                        'valid={:.4f},  best_valid_metric={:.4f}\n- DOA    :       -       '
                        'valid={:.4f},  best_valid_doa={:.4f}'.format(
                            ep, train_loss, valid_loss, self.best_valid_loss, valid_metric, self.config['valid_metric'],
                            self.best_valid_metric,
                            valid_doa, self.best_valid_doa))

                    if ep - self.best_epoch >= patience:
                        break

        self.model.load_state_dict(self.best_model_params)

    def _verbose_train_early_stopping_delta_error(self, train_loader, valid_loader, valid_data, optimizer, scheduler,
                                                  scaler):
        epochs = self.config['num_epochs']
        eval_freq = self.config['eval_freq']
        patience = self.config['patience']
        device = self.config['device']

        for ep in range(epochs):
            for _, data_batch in tqdm(enumerate(train_loader), total=len(train_loader),
                                      disable=self.config['disable_tqdm']):
                user_ids, item_ids, dim_ids, labels = data_batch

                user_ids = user_ids.to(device, non_blocking=True)
                item_ids = item_ids.to(device, non_blocking=True)
                dim_ids = dim_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self._compute_loss(user_ids, item_ids, dim_ids, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            # Early stopping
            if (ep + 1) % eval_freq == 0:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    valid_loss, valid_metric, valid_mae = self.evaluate_valid(valid_loader, valid_data.log_tensor)

                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")

                    # Checking loss improvement
                    if self.metric_sign * (self.best_valid_metric - valid_metric) / abs(
                            self.best_valid_metric) > 0.0001 or (self.best_valid_mae - valid_mae) / abs(
                            self.best_valid_mae) > 0.0001:
                        self.best_epoch = ep
                        self.best_valid_metric = valid_metric
                        self.best_valid_mae = valid_mae
                        self.best_model_params = self.model.state_dict()

                        scheduler.step(valid_loss)

                    if ep - self.best_epoch >= patience:
                        break
        self.model.load_state_dict(self.best_model_params)

    def _train_no_early_stopping(self, train_loader, valid_loader, valid_data, optimizer, scheduler, scaler):
        epochs = self.config['num_epochs']
        device = self.config['device']

        for _, _ in tqdm(enumerate(range(epochs + 1)), total=epochs, disable=self.config['disable_tqdm']):
            for data_batch in train_loader:
                user_ids = data_batch[:, 0].long()
                item_ids = data_batch[:, 1].long()
                labels = data_batch[:, 2]
                dim_ids = data_batch[:, 3].long()

                optimizer.zero_grad()

                with torch.amp.autocast('cuda'):
                    loss = self._compute_loss(user_ids, item_ids, dim_ids, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

    def _setup_params_paths(self):
        params_path = f"{self.config['params_path']}_{self.name}_fold_{self.fold}_seed_{self.config['seed']}.pth"
        emb_path = f"{self.config['embs_path']}_{self.name}_fold_{self.fold}_seed_{self.config['seed']}.csv"

        # if os.path.isfile(params_path) or os.path.isfile(emb_path):
        #     self._ask_saving_pref()
        # else:
        #     logging.info(params_path)
        #     logging.info(emb_path)

    def _tensorboard_saving(self, train_loss, valid_loss, valid_metric, ep):
        self.writer.add_scalars(f'{self.name}_{self.timestamp}_Loss',
                                {f'Train_{self.fold}': train_loss, f'Valid_{self.fold}': valid_loss}, ep)
        self.writer.add_scalars(f'{self.name}_{self.timestamp}_RMSE', {f'Valid_{self.fold}': valid_metric}, ep)

    def _flush_tensorboard_saving(self, train_loss, valid_loss, valid_metric, valid_mae, ep):
        self.writer.add_scalars(f'DBPR_Loss',
                                {f'Train_{self.name}': train_loss, f'Valid_{self.name}': valid_loss}, ep)
        self.writer.add_scalars(f'DBPR_pred',
                                {f'Valid_rmse_{self.name}': valid_metric, f'Valid_mae_{self.name}': valid_mae}, ep)
        self.writer.flush()

    def init_model(self, train_data: Dataset, valid_data: Dataset):

        if self.config['load_params']:
            self._load_model_params(temporary=False)
        self.state = "model_initialized"

    def get_graph(self):
        if self.state == "model_initialized":
            dummy_input = torch.tensor([[0, 0, 0], [0, 1, 2]])
            self.writer.add_graph(self.model, dummy_input)

    @property
    def name(self):
        return f'{self._name}_cont_model'

    @name.setter
    def name(self, new_value):
        self._name = new_value

    def _compute_loss(self, user_ids, item_ids, dim_ids, labels):
        pred = self.model(user_ids, item_ids, dim_ids)
        loss = self._loss_function(pred, labels)
        return loss

    @abstractmethod
    def _loss_function(self, pred, real):
        raise NotImplementedError

    def evaluation_state(func):
        """
        Temporary set the model state to "eval"
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract 'self' from the first positional argument
            self = args[0] if args else None
            if self is None:
                raise ValueError("Decorator 'evaluation_state' requires to be used on instance methods.")

            # Store the previous state
            prev_state = getattr(self, 'state', None)

            try:
                # Set the state to 'eval' before method execution
                self.state = "eval"
                # Call the actual method
                self.model.eval()
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    result = func(*args, **kwargs)
            finally:
                # Restore the previous state after method execution
                self.model.train()
                self.state = prev_state

            return result

        return wrapper

    @evaluation_state
    def _evaluate_preds(self, data_loader: data.DataLoader):
        loss_list = []
        pred_list = []
        label_list = []

        for data_batch in data_loader:
            user_ids = data_batch[:, 0].long()
            item_ids = data_batch[:, 1].long()
            labels = data_batch[:, 2]
            dim_ids = data_batch[:, 3].long()

            preds = self.model(user_ids, item_ids, dim_ids)

            loss = self._loss_function(preds, labels).float()
            loss_list.append(loss)
            pred_list.append(preds.detach())
            label_list.append(labels.detach())

        # Concatenate lists into tensors
        loss_tensor = torch.stack(loss_list)  # Assumes each loss is a scalar tensor
        pred_tensor = torch.cat(pred_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0)

        return loss_tensor, pred_tensor, label_tensor

    def _save_user_emb(self) -> None:
        path = self.config['embs_path'] + '_' + self.name + '_fold_' + str(self.fold) + '_seed_' + str(
            self.config['seed']) + ".csv"
        pd.DataFrame(self.get_user_emb().cpu().numpy()).to_csv(path, index=None, header=None)

    def _save_model_params(self, temporary=True) -> None:
        path = self.config['params_path'] + '_' + self.name + '_fold_' + str(self.fold) + '_seed_' + str(
            self.config['seed'])
        if temporary:
            path += '_temp'

        torch.save(self.model.state_dict(), path + '.pth')

    def _delete_temp_model_params(self) -> None:
        path = self.config['params_path'] + '_' + self.name + '_fold_' + str(self.fold) + '_seed_' + str(
            self.config['seed']) + '_temp.pth'
        os.remove(path)

    def _load_model_params(self, temporary=True) -> None:
        path = self.config['params_path'] + '_' + self.name + '_fold_' + str(self.fold) + '_seed_' + str(
            self.config['seed'])
        if temporary:
            path += '_temp'
        self.model.load_state_dict(torch.load(path + '.pth', map_location=torch.device(self.config['device'])))
        self.model.to(self.config['device'])

    def _ask_saving_pref(self):
        # Print the prompt to the terminal
        logging.info(
            "-- Some parameters have already been saved for your model on the same dataset with the same seed -- ")
        logging.info("Do you want to overwrite them?")
        logging.info("yes [y/Y]' or 'no [n/N]")
        sys.stdout.flush()

        # Get user input
        user_input = input().strip().lower()

        # Process the input
        if "y" in user_input.lower():
            self._overwrite_button()
        elif "n" in user_input.lower():
            self._dont_overwrite_button()
        else:
            logging.info("Invalid input. Please enter 'yes (y/Y)' or 'no (n/N)'.")
            self._ask_saving_pref()  # Retry if invalid input

    def _overwrite_button(self):
        logging.info("You chose to overwrite the saved parameters.")
        # Perform any further logic here

    def _dont_overwrite_button(self):
        self.config["save_params"] = False
        logging.info("You chose not to overwrite the saved parameters.")
        # Perform any further logic here

    def _ask_loading_pref(self):
        # Print the prompt to the terminal
        logging.info(
            "-- Are you sure you want to load previously learned model parameters \n instead of learning them again? -- ")
        logging.info("yes [y/Y]' or 'no [n/N]")
        sys.stdout.flush()

        # Get user input
        user_input = input().strip().lower()

        # Process the input
        if "y" in user_input.lower():
            self._load_button()
        elif "n" in user_input.lower():
            self._dont_load_button()
        else:
            logging.info("Invalid input. Please enter 'yes (y/Y)' or 'no (n/N)'.")
            self._ask_loading_pref()  # Retry if invalid input

    def _load_button(self):
        logging.info("You chose to load previously saved parameters.")
        # Perform any further logic here

    def _dont_load_button(self):
        self.config["load_params"] = False
        logging.info("You chose not to load previously saved parameters.")

    def get_user_emb(self):
        if not self._trained:
            warnings.warn("The model must be trained before getting user embeddings")
        return None

    def evaluate_valid(self, valid_dataloader: data.DataLoader, log_tensor):
        loss_list, pred_list, label_list = self._evaluate_preds(valid_dataloader)

        return torch.mean(torch.tensor(loss_list, dtype=torch.float)), root_mean_squared_error(
            torch.tensor(pred_list, dtype=torch.float), torch.tensor(label_list, dtype=torch.float)), pred_list

    def evaluate_predictions(self, test_dataset: data.DataLoader):
        test_dataloader = data.DataLoader(test_dataset, batch_size=100000, shuffle=False)
        loss_tensor, pred_tensor, label_tensor = self._evaluate_preds(test_dataloader)
        # Convert tensors to double if needed
        pred_tensor = pred_tensor.double()
        label_tensor = label_tensor.double()

        # Compute metrics in one pass using a dictionary comprehension
        results = {metric: self.pred_metric_functions[metric](pred_tensor, label_tensor).cpu().item()
                   for metric in self.pred_metrics}

        # Optionally keep the predictions and labels as tensors to avoid conversion overhead
        results.update({
            'preds': pred_tensor,
            'labels': label_tensor
        })

        return results

    @evaluation_state
    def evaluate_profiles(self, dataloader: dataset.LoaderDataset):
        emb = self.get_user_emb()

        results = {metric: self.profile_metric_functions[metric](emb, dataloader)
                   for metric in self.profile_metrics}

        return results


def compute_pc_er(emb, test_data):
    U_resp_sum = torch.zeros(size=(test_data.n_users, test_data.n_categories)).to(test_data.raw_data_array.device,
                                                                                  non_blocking=True)
    U_resp_nb = torch.zeros(size=(test_data.n_users, test_data.n_categories)).to(test_data.raw_data_array.device,
                                                                                 non_blocking=True)

    data_loader = data.DataLoader(test_data, batch_size=1, shuffle=False)
    for data_batch in data_loader:
        user_ids = data_batch[:, 0].long()
        item_ids = data_batch[:, 1].long()
        labels = data_batch[:, 2]
        dim_ids = data_batch[:, 3].long()

        U_resp_sum[user_ids, dim_ids] += labels
        U_resp_nb[user_ids, dim_ids] += torch.ones_like(labels)

    U_ave = U_resp_sum / U_resp_nb

    return pc_er(test_data.n_categories, U_ave, emb).cpu().item()


@torch.jit.script
def pc_er(concept_n: int, U_ave: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
    """
    Compute the average correlation across dimensions between U_ave and emb.
    Both U_ave and emb should be [num_user, concept_n] tensors.
    concept_n: number of concepts
    U_ave: FloatTensor [num_user, concept_n]
    emb: FloatTensor [num_user, concept_n]

    Returns:
        c: A 0-dim tensor (scalar) representing the average correlation.
    """

    # Initialize counters as tensors to allow JIT compatibility
    c = torch.tensor(0.0, device=U_ave.device)
    s = torch.tensor(0.0, device=U_ave.device)

    for dim in range(concept_n):
        mask = ~torch.isnan(U_ave[:, dim])
        U_dim_masked = U_ave[:, dim][mask].unsqueeze(1)
        Emb_dim_masked = emb[:, dim][mask].unsqueeze(1)

        corr = torch.corrcoef(torch.concat([U_dim_masked, Emb_dim_masked], dim=1).T)[0][1]

        if not torch.isnan(corr):
            c += corr
            s += 1
    c /= s
    return c


def compute_rm(emb:torch.Tensor, test_data):
    concept_array, concept_lens = utils.preprocess_concept_map(test_data.concept_map)
    return utils.compute_rm_fold(emb.cpu().numpy(), test_data.raw_data, concept_array, concept_lens)


def compute_doa(emb:torch.Tensor, test_data):
    return np.mean(utils.evaluate_doa(emb.cpu().numpy(), test_data.log_tensor.cpu().numpy(), test_data.metadata,
                                      test_data.concept_map))

@torch.jit.script
def root_mean_squared_error(y_true, y_pred):
    """
    Compute the rmse metric (Regression)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The rmse metric.
    """
    return torch.sqrt(torch.mean(torch.square(y_true - y_pred)))


@torch.jit.script
def mean_absolute_error(y_true, y_pred):
    """
    Compute the mae metric (Regression)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The mae metric.
    """
    return torch.mean(torch.abs(y_true - y_pred))


@torch.jit.script
def r2(gt, pd):
    """
    Compute the r2 metric (Regression)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The r2 metric.
    """

    mean = torch.mean(gt)
    sst = torch.sum(torch.square(gt - mean))
    sse = torch.sum(torch.square(gt - pd))

    r2 = 1 - sse / sst

    return r2


@torch.jit.script
def micro_ave_accuracy(y_true, y_pred):
    """
    Compute the micro-averaged accuracy (Classification)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The micro-averaged precision.
    """
    return torch.mean((y_true == y_pred).float())


@torch.jit.script
def micro_ave_precision(y_true, y_pred):
    """
    Compute the micro-averaged precision (Binary classification)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The micro-averaged precision.
    """
    true_positives = torch.sum((y_true == 2) & (y_pred == 2)).float()
    predicted_positives = torch.sum(y_pred == 2).float()
    return true_positives / predicted_positives


@torch.jit.script
def micro_ave_recall(y_true, y_pred):
    """
    Compute the micro-averaged recall (Binary classification)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The micro-averaged recall.
    """
    true_positives = torch.sum((y_true == 2) & (y_pred == 2)).float()
    actual_positives = torch.sum(y_true == 2).float()
    return true_positives / actual_positives


@torch.jit.script
def micro_ave_f1(y_true, y_pred):
    """
    Compute the micro-averaged f1 (Binary classification)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The micro-averaged f1.
    """
    precision = micro_ave_precision(y_true, y_pred)
    recall = micro_ave_recall(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)


def micro_ave_auc(y_true, y_pred):
    """
    Compute the micro-averaged roc-auc (Binary classification)

    Args:
        y_true (Tensor): Ground truth labels.
        y_pred (Tensor): Predicted labels.

    Returns:
        Tensor: The micro-averaged roc-auc.
    """
    y_true = y_true.cpu().int().numpy()
    y_pred = y_pred.cpu().int().numpy()
    roc_auc = roc_auc_score(y_true.ravel(), y_pred.ravel(), average='micro')
    return torch.tensor(roc_auc)
