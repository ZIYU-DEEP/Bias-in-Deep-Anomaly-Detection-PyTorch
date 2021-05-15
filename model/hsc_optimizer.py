from base_optimizer import BaseTrainer, BaseEvaluater
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import torch.nn as nn
import numpy as np
import logging
import torch
import time


# #########################################################################
# 1. OC Trainer (Only use the Encoder Network as Net)
# #########################################################################
class HSCTrainer(BaseTrainer):
    def __init__(self,
                 optimizer_name: str = 'adam',
                 lr: float = 0.01,
                 n_epochs: int = 60,
                 lr_milestones: tuple = (50, 100, 150, 200),
                 batch_size: int = 128,
                 weight_decay: float = 1e-6,
                 device: str = 'cuda:1',
                 n_jobs_dataloader: int = 0):

        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size,
                         weight_decay, device, n_jobs_dataloader)
        self.eps = 1e-6
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset, net, label_normal, split=False):
        # Get the logger
        logger = logging.getLogger()

        if split:
            train_loader, _ = dataset.loaders(batch_size=self.batch_size,
                                              num_workers=self.n_jobs_dataloader)
        else:
            train_loader = dataset.loaders(batch_size=self.batch_size,
                                           num_workers=self.n_jobs_dataloader)

        logger.info('Hey I am loading net for you!')
        net = net.to(self.device)

        logger.info('Setting hyper-parameters!')
        criterion = nn.MSELoss(reduction='none')
        optimizer = optim.Adam(net.parameters(),
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.lr_milestones,
                                                   gamma=0.1)

        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' %
                            float(scheduler.get_lr()[0]))

            epoch_loss, n_batches = 0.0, 0
            epoch_start_time = time.time()

            for data in train_loader:
                inputs, y, _ = data
                inputs, y = inputs.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = net(inputs)
                # dist = torch.sum((outputs - self.c) ** 2, dim=1)
                dist = torch.norm(outputs, p=2, dim=1) ** 2
                scores = 1 - torch.exp(- dist)
                # The following is the core loss function
                # We assume here normal label is (0,) and abnormal label is (1,)
                losses = torch.where(torch.tensor(np.isin(y.cpu().data.numpy(), label_normal)).to(self.device),
                                     dist,
                                     - torch.log(scores + self.eps))
                loss = torch.mean(losses)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | '
                        f'Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')
        return net

    def test(self, dataset, net, label_normal, split=False):
        # Get the logger
        logger = logging.getLogger()

        if split:
            _, test_loader = dataset.loaders(batch_size=self.batch_size,
                                             num_workers=self.n_jobs_dataloader)
        else:
            test_loader = dataset.loaders(batch_size=self.batch_size,
                                          num_workers=self.n_jobs_dataloader)

        net = net.to(self.device)

        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, y, idx = data
                inputs, y, idx = inputs.to(self.device), y.to(self.device), idx.to(self.device)

                outputs = net(inputs)
                # dist = torch.sum((outputs - self.c) ** 2, dim=1)
                dist = torch.norm(outputs, p=2, dim=1) ** 2
                scores = 1 - torch.exp(-dist)
                losses = torch.where(torch.tensor(np.isin(y.cpu().data.numpy(), label_normal)).to(self.device),
                                     dist,
                                     - torch.log(scores + self.eps))
                loss = torch.mean(losses)
                # scores = dist
                scores = 1 - torch.exp(-dist)

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            y.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)
        self.test_auc = roc_auc_score(np.isin(labels, label_normal), scores)

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')


# #########################################################################
# 2. Evaluater
# #########################################################################
class HSCEvaluater(BaseEvaluater):
    def __init__(self,
                 batch_size: int = 32,
                 device: str = 'cuda:1',
                 n_jobs_dataloader: int = 0):
        super().__init__(batch_size, device, n_jobs_dataloader)

        # Hyper-parameter for the weight of anomaly training
        self.eps = 1e-6

        # Results
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def test(self,
             dataset,
             net,
             label_normal):
        # Get test data loader
        all_loader = dataset.loaders(batch_size=self.batch_size,
                                     num_workers=self.n_jobs_dataloader)

        # Set device for network
        net = net.to(self.device)

        # Set loss
        criterion = nn.MSELoss(reduction='none')

        # Testing
        print('Starting evaluating...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in all_loader:
                inputs, y, idx = data
                inputs, y, idx = inputs.to(self.device), y.to(self.device), idx.to(self.device)

                outputs = net(inputs)
                # dist = torch.sum((outputs - self.c) ** 2, dim=1)
                dist = torch.norm(outputs, p=2, dim=1) ** 2
                scores = 1 - torch.exp(-dist)

                losses = torch.where(torch.tensor(np.isin(y.cpu().data.numpy(), label_normal)).to(self.device),
                                     dist,
                                     - torch.log(scores + self.eps))

                loss = torch.mean(losses)
                scores = 1 - torch.exp(-dist)

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            y.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        # Log results
        print('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        print('Test Time: {:.3f}s'.format(self.test_time))
        print('Finished testing.')
