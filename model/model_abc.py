import sys
sys.path.append('../network/')

import torch
import json
from network.main import build_network
from .abc_optimizer import ABCTrainer, ABCEvaluater


# --------------------------------------------
# 3.3. (a) Model Object for training
# --------------------------------------------
class ABCModel:
    def __init__(self):
        self.net_name = None
        self.net = None
        self.trainer = None
        self.optimizer_name = None
        self.results = {'train_time': None, 'test_auc': None,
                        'test_time': None, 'test_scores': None}

    def set_network(self, net_name, rep_dim, x_dim, h_dims, bias):
        self.net_name = net_name
        self.net = build_network(net_name, rep_dim, x_dim, h_dims, bias)

    def train(self, dataset, eta=None, optimizer_name: str = 'adam', lr: float = 0.001,
              n_epochs: int = 60, lr_milestones: tuple = (100, 160, 220),
              batch_size: int = 32, weight_decay: float = 1e-6, device: str = 'cuda:1',
              n_jobs_dataloader: int = 0, label_normal: tuple=(0,), split: bool=False):
        print('Learning rate: {}'.format(lr))

        self.trainer = ABCTrainer(optimizer_name, lr, n_epochs, lr_milestones,
                                  batch_size, weight_decay, device, n_jobs_dataloader)

        self.net = self.trainer.train(dataset, self.net, label_normal, split)
        self.results['train_time'] = self.trainer.train_time

    def test(self,
             dataset,
             device: str = 'cuda:1',
             n_jobs_dataloader: int = 0,
             label_normal: tuple=(0,),
             split: bool=False):
        self.trainer.test(dataset, self.net, label_normal, split)
        if self.trainer.test_auc:
            self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def save_model(self, export_model, save_ae=True):
        net_dict = self.net.state_dict()
        torch.save({'net_dict': net_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        model_dict = torch.load(model_path, map_location=map_location)
        self.net.load_state_dict(model_dict['net_dict'])

    def save_results(self, export_json):
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)


# --------------------------------------------
# 3.3. (b) Model Object for Evaluating
# --------------------------------------------
class ABCModelEval:
    def __init__(self):
        self.net = None
        self.evaluater= None
        self.optimizer_name = None
        self.results = {'test_time': None,'test_scores': None}

    def set_network(self, net_name, rep_dim, x_dim, h_dims, bias):
        self.net_name = net_name
        self.net = build_network(net_name, rep_dim, x_dim, h_dims, bias)

    def load_model(self, model_path, map_location='cuda:1'):
        model_dict = torch.load(model_path, map_location=map_location)
        self.net.load_state_dict(model_dict['net_dict'])

    def test(self,
             dataset,
             eta=None,
             batch_size: int = 32,
             device: str = 'cuda:1',
             n_jobs_dataloader: int = 0,
             label_normal: tuple=(0,)):

        if self.evaluater is None:
            self.evaluater = ABCEvaluater(batch_size, device, n_jobs_dataloader)

        self.evaluater.test(dataset, self.net, label_normal)
        self.results['test_time'] = self.evaluater.test_time
        self.results['test_scores'] = self.evaluater.test_scores

    def save_results(self, export_json):
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
