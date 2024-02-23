import copy
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from scipy.special import expit
from tdc import Evaluator

from .dataloaders import FastDataLoader, InfiniteDataLoader
from .utils import prepare_data, forward_pass, get_collate_functions, mean_rprecision


class BaseTrainer:
    def __init__(self, args, dataset, network, criterion, optimizer, scheduler):
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        # Dataset settings
        self.train_dataset = dataset
        self.train_dataset.mode = 0
        self.eval_dataset = copy.deepcopy(dataset)
        self.eval_dataset.mode = 2
        self.num_tasks = dataset.num_tasks
        self.train_collate_fn, self.eval_collate_fn = get_collate_functions(args, self.train_dataset)

        # Training hyperparameters
        self.args = args
        self.train_update_iter = args.train_update_iter
        self.stride = args.stride
        self.max_length = self.network.model.config.n_positions
        self.mini_batch_size = args.mini_batch_size
        self.num_workers = args.num_workers
        self.base_trainer_str = self.get_base_trainer_str()

        # Evaluation and metrics
        self.split_time = args.split_time
        self.test_time = args.test_time
        self.eval_next_timestamps = args.eval_next_timestamps
        self.task_accuracies = {}
        self.worst_time_accuracies = {}
        self.best_time_accuracies = {}
        self.eval_metric = 'accuracy'

        self.best_val = None
        self.patience = args.patience
        self.current_patience = args.patience
        self.eval_freq = args.eval_freq
        # if str(self.eval_dataset) == 'drug':
        #     self.eval_metric = 'PCC'
        # elif 'mimic' in str(self.eval_dataset) and self.args.prediction_type == 'mortality':
        #     self.eval_metric = 'ROC-AUC'

    def __str__(self):
        pass

    def get_base_trainer_str(self):
        base_trainer_str = f'train_update_iter={self.train_update_iter}-lr={self.args.lr}-' \
                           f'mini_batch_size={self.args.mini_batch_size}-seed={self.args.random_seed}'
        if self.args.eval_fix:
            base_trainer_str += f'-eval_fix'
        else:
            base_trainer_str += f'-eval_stream'
        return base_trainer_str

    def train_step(self, dataloader, timestamp):
        self.network.train()
        loss_all = []
        for step, (x, _) in enumerate(dataloader):
            if self.current_patience == 0:
                print('Stopped at step ' + str(step-1))
                if self.scheduler is not None:
                    self.scheduler.step()
                break
            x = torch.stack(x)
            x = prepare_data(x)
            logit, loss = forward_pass(x, self.network)
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # EVAL
            if step % self.eval_freq == 0 and step > 0:
                self.network.eval()
                val_loss_all = []
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)

                test_time_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                      batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                for _, sample in enumerate(test_time_dataloader):
                    if len(sample) == 3:
                        x, _, _ = sample
                    else:
                        x, _ = sample
                    x = torch.stack(x)
                    x = prepare_data(x)
                    with torch.no_grad():
                        logit, loss = forward_pass(x, self.network)
                        val_loss_all.append(loss.item())

                val_loss = np.mean(val_loss_all)
                print(f'Val loss epoch {step/self.eval_freq}: {val_loss}')
                self.eval_dataset.mode = 2
                self.network.train()

                # EARLY STOPPING
                if self.best_val is None:
                    is_best = True
                else:
                    is_best = val_loss < self.best_val
                if is_best or step < 3 * self.eval_freq:
                    # Reset patience
                    self.current_patience = self.patience
                    self.best_val = val_loss
                    self.save_model(timestamp)
                else:
                    # Decrease patience
                    self.current_patience -= 1

            if step == self.train_update_iter:
                if self.scheduler is not None:
                    self.scheduler.step()
                break

    def train_online(self):
        for i, timestamp in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.eval_fix and timestamp == self.split_time:
                break
            if self.args.load_model and self.model_path_exists(timestamp):
                self.load_model(timestamp)
            else:
                if self.args.lisa and i == self.args.lisa_start_time:
                    self.lisa = True
                self.train_dataset.update_current_timestamp(timestamp)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                      batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.save_model(timestamp)
                self.train_step(train_dataloader, timestamp)
                path = self.get_model_path(timestamp)
                print(f'Saved model at timestamp {timestamp} to path {path}...\n')
                # Reset patience
                self.current_patience = self.patience
                if self.args.method in ['coral', 'groupdro', 'irm', 'erm']:
                    self.train_dataset.update_historical(i + 1, data_del=True)

    def train_offline(self):
        if self.args.method in ['simclr', 'swav']:
            self.train_dataset.ssl_training = True
        for i, timestamp in enumerate(self.train_dataset.ENV):
            if timestamp < self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                self.train_dataset.update_historical(i + 1)
                self.train_dataset.mode = 1
                self.train_dataset.update_current_timestamp(timestamp)
                self.train_dataset.update_historical(i + 1, data_del=True)
            elif timestamp == self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                if self.args.load_model:
                    self.load_model(timestamp)
                else:
                    self.save_model(timestamp)
                    self.train_step(train_id_dataloader, timestamp)
                    path = self.get_model_path(timestamp)
                    print(f'Saved model at timestamp {timestamp} to path {path}...\n')
                    # Reset patience
                    self.current_patience = self.patience
                break

    def network_evaluation(self, test_time_dataloader):
        self.network.eval()
        loss_all = []
        for _, sample in enumerate(test_time_dataloader):
            if len(sample) == 3:
                x, _, _ = sample
            else:
                x, _ = sample
            x = torch.stack(x)
            x = prepare_data(x)
            with torch.no_grad():
                logit, loss = forward_pass(x, self.network)
                loss_all.append(loss.item())

        loss = torch.tensor(loss_all)
        avg_loss = loss.mean()
        ppl = torch.exp(avg_loss)
        metric = (ppl, avg_loss)

        self.network.train()

        return metric, loss

    def evaluate_stream(self, start):
        self.network.eval()
        metric_all = []
        for i in range(start, min(start + self.eval_next_timestamps, len(self.eval_dataset.ENV))):
            test_time = self.eval_dataset.ENV[i]
            self.eval_dataset.update_current_timestamp(test_time)
            test_time_dataloader = FastDataLoader(dataset=self.eval_dataset, batch_size=self.mini_batch_size,
                                                  num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            metric, _ = self.network_evaluation(test_time_dataloader)
            metric_all.append(metric)

        avg_metric, worst_metric, best_metric = np.mean(metric_all, axis=0), np.min(metric_all, axis=0), np.max(
            metric_all, axis=0)

        print(
            f'Timestamp = {self.eval_dataset.ENV[start]}'
            f'\t Average {self.eval_metric}: {avg_metric}'
            f'\t Worst {self.eval_metric}: {worst_metric}'
            f'\t Best {self.eval_metric}: {best_metric}'
            f'\t Performance over all timestamps: {metric_all}'
        )
        self.network.train()

        return avg_metric, worst_metric, best_metric

    def evaluate_online(self):
        print(f'\n=================================== Results (Eval-Stream) ===================================')
        print(f'Metric: {self.eval_metric}')
        end = len(self.eval_dataset.ENV) - self.eval_next_timestamps
        for i, timestamp in enumerate(self.eval_dataset.ENV[:end]):
            self.load_model(timestamp)
            if timestamp == self.args.split_time:
                self.evaluate_online_all_timestamps()
            avg_metric, worst_metric, best_metric = self.evaluate_stream(i + 1)
            self.task_accuracies[timestamp] = avg_metric
            self.worst_time_accuracies[timestamp] = worst_metric
            self.best_time_accuracies[timestamp] = best_metric

    def evaluate_online_all_timestamps(self):
        print(f'\n=================================== Results (Eval-Fix) ===================================')
        timestamps = self.train_dataset.ENV
        metric_all, loss_all = [], []
        for i, timestamp in enumerate(timestamps):
            if timestamp <= self.split_time:
                pass
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                metric, loss = self.network_evaluation(test_ood_dataloader)
                loss_all.append(loss)

                print(f'OOD timestamp = {timestamp}: \t {self.eval_metric} is {metric}')
                metric_all.append(metric)
        print(f'\nAverage Metric Across All Timestamps: \t{np.mean(metric_all, axis=0)}'
              f'\nWorst Metric Across All Timestamps: \t{np.min(metric_all, axis=0)}'
              f'\nMetrics Across All Timestamps: \t{metric_all}')

        loss = torch.tensor(loss_all).mean()
        ppl = torch.exp(loss)
        print(f'\nMetric Across All Timestamps: \t{[ppl, loss]}')

    def evaluate_offline(self):
        print(f'\n=================================== Results (Eval-Fix) ===================================')
        print(f'Metric: {self.eval_metric}')
        timestamps = self.eval_dataset.ENV
        metric_all, loss_all = [], []
        for i, timestamp in enumerate(timestamps):
            if timestamp < self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                self.eval_dataset.update_historical(i + 1, data_del=True)
            elif timestamp == self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                id_metric, _ = self.network_evaluation(test_id_dataloader)
                print(f'ID {self.eval_metric}: \t{id_metric}\n')
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                metric, loss = self.network_evaluation(test_ood_dataloader)
                print(f'OOD timestamp = {timestamp}: \t {self.eval_metric} is {metric}')
                metric_all.append(metric)
                if timestamp > self.test_time:
                    loss_all.append(loss)
        print(f'\nOOD Average Metric: \t{np.mean(metric_all, axis=0)}'
              f'\nOOD Worst Metric: \t{np.min(metric_all, axis=0)}'
              f'\nAll OOD Metrics: \t{metric_all}')

        loss = torch.tensor(loss_all).mean()
        ppl = torch.exp(loss)
        print(f'\nMetric Across All Timestamps: \t{[ppl, loss]}')

    def evaluate_offline_all_timestamps(self):
        print(f'\n=================================== Results (Eval-Fix) ===================================')
        timestamps = self.train_dataset.ENV
        metric_all, loss_all = [], []
        for i, timestamp in enumerate(timestamps):
            if timestamp <= self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                metric, _ = self.network_evaluation(test_id_dataloader)
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                metric, loss = self.network_evaluation(test_ood_dataloader)
                loss_all.append(loss)
            print(f'OOD timestamp = {timestamp}: \t {self.eval_metric} is {metric}')
            metric_all.append(metric)
        print(f'\nAverage Metric Across All Timestamps: \t{np.mean(metric_all, axis=0)}'
              f'\nWorst Metric Across All Timestamps: \t{np.min(metric_all, axis=0)}'
              f'\nMetrics Across All Timestamps: \t{metric_all}')

        loss = torch.tensor(loss_all).mean()
        ppl = torch.exp(loss)
        print(f'\nMetric Across All Timestamps: \t{[ppl, loss]}')

    def run_eval_fix(self):
        print('==========================================================================================')
        print("Running Eval-Fix...")
        if self.args.method in ['er', 'agem', 'ewc', 'ft', 'si']:
            self.train_online()
        else:
            self.train_offline()
        if self.args.eval_all_timestamps:
            self.evaluate_offline_all_timestamps()
        else:
            self.evaluate_offline()

    def run_task_difficulty(self):
        print('==========================================================================================')
        print("Running Task Difficulty...")
        timestamps = self.train_dataset.ENV
        metric_all = []
        for i, timestamp in enumerate(timestamps):
            self.train_dataset.mode = 0
            self.train_dataset.update_current_timestamp(timestamp)
            if i < len(timestamps) - 1:
                self.train_dataset.update_historical(i + 1)
            else:
                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                if self.args.load_model:
                    self.load_model(timestamp)
                else:
                    self.train_step(train_id_dataloader, timestamp)
                    self.save_model(timestamp)

        for i, timestamp in enumerate(timestamps):
            self.eval_dataset.mode = 1
            self.eval_dataset.update_current_timestamp(timestamp)
            test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                 batch_size=self.mini_batch_size,
                                                 num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            metric, _, _ = self.network_evaluation(test_ood_dataloader)
            metric = round(metric, 2)
            print(f'OOD timestamp = {timestamp}: \t {self.eval_metric} is {metric}')
            metric_all.append(metric)
        print(f'Average Metric: {np.mean(metric_all, axis=0)}')
        print(f'Worst timestamp accuracy: {np.min(metric_all, axis=0)}')
        print(f'All timestamp accuracies: {metric_all}')

    def run_eval_stream(self):
        print('==========================================================================================')
        print("Running Eval-Stream...")
        if not self.args.load_model:
            self.train_online()
        self.evaluate_online()

    def run(self):
        torch.cuda.empty_cache()
        start_time = time.time()
        if self.args.difficulty:
            self.run_task_difficulty()
        elif self.args.eval_fix:
            self.run_eval_fix()
        else:
            self.run_eval_stream()
        runtime = time.time() - start_time
        print(f'Runtime: {runtime:.2f}\n')

    def get_model_path(self, timestamp):
        model_str = f'{str(self.train_dataset)}_{str(self)}_time={timestamp}'
        path = os.path.join(self.args.log_dir, model_str)
        return path

    def model_path_exists(self, timestamp):
        return os.path.exists(self.get_model_path(timestamp))

    def save_model(self, timestamp):
        path = self.get_model_path(timestamp)
        torch.save(self.network.state_dict(), path)
        # print(f'Saved model at timestamp {timestamp} to path {path}...\n')

    def load_model(self, timestamp):
        path = self.get_model_path(timestamp)
        self.network.load_state_dict(torch.load(path), strict=False)
