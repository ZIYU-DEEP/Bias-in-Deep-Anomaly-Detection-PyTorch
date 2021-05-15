"""
Title: base_optimizer.py
Description: The base optimizer.
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/optim
"""

from abc import ABC, abstractmethod


# #########################################################################
# 1. Base Trainer
# #########################################################################
class BaseTrainer(ABC):
    def __init__(self,
                 optimizer_name: str,
                 lr: float,
                 n_epochs: int,
                 lr_milestones: tuple,
                 batch_size: int,
                 weight_decay: float,
                 device: str,
                 n_jobs_dataloader: int):

        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.n_epochs = n_epochs
        self.lr_milestones = lr_milestones
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader

    @abstractmethod
    def train(self, dataset, net):
        pass

    @abstractmethod
    def test(self, dataset, net):
        pass


# #########################################################################
# 2. Base Evaluater
# #########################################################################
class BaseEvaluater(ABC):
    def __init__(self,
                 batch_size: int,
                 device: str,
                 n_jobs_dataloader: int):

        super().__init__()
        self.batch_size = batch_size
        self.device = device
        self.n_jobs_dataloader = n_jobs_dataloader


    @abstractmethod
    def test(self, dataset, net):
        pass
