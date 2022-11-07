import sys
sys.path.append('../network/')

import torch
import json
from network.main import build_network, build_autoencoder
from .one_class_optimizer import OneClassTrainer, OneClassTrainer_, OneClassEvaluater, AETrainer


# #########################################################################
# 1. Model Object for Training
# #########################################################################
class OneClassModel:
    def __init__(self,
                 optimizer_: str = 'one_class',
                 eta: float = 1.0):
        known_optimizer_ = ('one_class', 'one_class_unsupervised')
        assert optimizer_ in known_optimizer_

        self.optimizer_ = optimizer_
        self.c = None
        self.eta = eta
        self.net_name = None

        self.net = None
        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None
        self.ae_trainer = None
        self.optimizer_name = None

        self.results = {'train_time': None, 'test_auc': None,
                        'test_time': None, 'test_scores': None}
        self.ae_results = {'train_time': None, 'test_auc': None,
                           'test_time': None}

    def set_network(self, net_name, rep_dim, x_dim, h_dims, bias):
        """
        Set the network structure for the model.
        The key here is to initialize <self.net>.
        """
        self.net_name = net_name
        self.net = build_network(net_name, rep_dim, x_dim, h_dims, bias)
        self.ae_net = build_autoencoder(net_name, rep_dim, x_dim, h_dims, bias)

    def load_model(self,
                   model_path,
                   load_ae=False,
                   map_location='cuda:1'):
        """
        Load the trained model for the model.
        The key here is to initialize <self.c>.
        """
        # Load the general model
        model_dict = torch.load(model_path, map_location=map_location)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

        # Load autoencoder parameters if specified
        if load_ae:
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])

    def init_network_weights_from_pretraining(self):
        """
        If pretraining is specified, we will load the networks
        from the pretrained ae net.
        """
        # Obtain the net dictionary
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}

        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)

        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def train(self,
              dataset,
              eta: float=1.0,
              optimizer_name: str='adam',
              lr: float=0.001,
              n_epochs: int=60,
              lr_milestones: tuple=(100, 160, 220),
              batch_size: int=32,
              weight_decay: float=1e-6,
              device: str='cuda:1',
              n_jobs_dataloader: int=0,
              label_normal: tuple=(0,),
              split: bool=False):
        print('Learning rate: {}'.format(lr))
        self.optimizer_name = optimizer_name

        if self.optimizer_ == 'one_class':
            self.trainer = OneClassTrainer(self.c,
                                           self.eta,
                                           optimizer_name,
                                           lr,
                                           n_epochs,
                                           lr_milestones,
                                           batch_size,
                                           weight_decay,
                                           device,
                                           n_jobs_dataloader)
        if self.optimizer_ == 'one_class_unsupervised':
            self.trainer = OneClassTrainer_(self.c,
                                           self.eta,
                                           optimizer_name,
                                           lr,
                                           n_epochs,
                                           lr_milestones,
                                           batch_size,
                                           weight_decay,
                                           device,
                                           n_jobs_dataloader)

        self.net = self.trainer.train(dataset, self.net, label_normal, split)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()

    def test(self,
             dataset,
             device: str = 'cuda:1',
             n_jobs_dataloader: int = 0,
             label_normal: tuple=(0,),
             split: bool=False):
        if self.trainer is None:
            if self.optimizer_ == 'one_class':
                self.trainer = OneClassTrainer(self.c,
                                               self.eta,
                                               device=device,
                                               n_jobs_dataloader=n_jobs_dataloader)
            if self.optimizer_ == 'one_class_unsupervised':
                self.trainer = OneClassTrainer_(self.c,
                                                self.eta,
                                                device=device,
                                                n_jobs_dataloader=n_jobs_dataloader)
        self.trainer.test(dataset, self.net, label_normal, split)

        if self.trainer.test_auc:
            self.results['test_auc'] = self.trainer.test_auc
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores

    def pretrain(self,
                 dataset,
                 optimizer_name: str='adam',
                 lr: float=0.001,
                 n_epochs: int=100,
                 lr_milestones: tuple=(50, 100, 150, 200),
                 batch_size: int=128,
                 weight_decay: float=1e-6,
                 device: str='cuda:1',
                 n_jobs_dataloader: int=0,
                 split_train: bool=False,
                 split_test: bool=False):

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name,
                                    lr=lr,
                                    n_epochs=n_epochs,
                                    lr_milestones=lr_milestones,
                                    batch_size=batch_size,
                                    weight_decay=weight_decay,
                                    device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net, split_train)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        self.ae_trainer.test(dataset, self.ae_net, split_test)

        # Get test results
        # self.ae_results['test_auc'] = self.ae_trainer.test_auc
        self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def save_model(self, export_model, save_ae=True):
        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict() if (save_ae and self.ae_net is not None) else None

        torch.save({'c': self.c,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def save_results(self, export_json):
        with open(export_json, 'w') as f:
            json.dump(self.results, f)

    def save_ae_results(self, export_json):
        with open(export_json, 'w') as f:
            json.dump(self.ae_results, f)


# #########################################################################
# 2. Model Object for Training
# #########################################################################
class OneClassModelEval:
    def __init__(self,
                 optimizer_,
                 eta: float = 1.0):
        known_optimizer_ = ('one_class', 'one_class_unsupervised')
        assert optimizer_ in known_optimizer_
        self.optimizer_ = optimizer_
        self.eta = eta
        self.net_name = None
        self.net = None
        self.evaluater= None
        self.optimizer_name = None
        self.results = {'test_time': None,'test_scores': None}

    def set_network(self, net_name, rep_dim, x_dim, h_dims, bias):
        """
        Set the network structure for the model.
        The key here is to initialize <self.net>.
        """
        self.net_name = net_name
        self.net = build_network(net_name, rep_dim, x_dim, h_dims, bias)

    def load_model(self, model_path, map_location='cuda:1'):
        """
        The key here is to fill in <self.c> and <self.net>.
        """
        model_dict = torch.load(model_path, map_location=map_location)
        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])

    def test(self,
             dataset,
             eta: float=1,
             batch_size: int=128,
             device: str='cuda:1',
             n_jobs_dataloader: int = 0,
             label_normal: tuple=(0,)):

        if self.evaluater is None:
            self.evaluater = OneClassEvaluater(self.c,
                                               self.eta,
                                               batch_size=batch_size,
                                               device=device,
                                               n_jobs_dataloader=n_jobs_dataloader)

        self.evaluater.test(self.optimizer_,
                            dataset,
                            self.net,
                            label_normal)
        self.results['test_time'] = self.evaluater.test_time
        self.results['test_scores'] = self.evaluater.test_scores

    def save_results(self, export_json):
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)
