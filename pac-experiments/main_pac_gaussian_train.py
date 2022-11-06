"""
[Title] main_pac_gaussian_train.py
[Description] The main file to train the synthetic dataset.
"""

# ############################################
# 0. Preparation
# ############################################
import sys
sys.path.append('../loader/')
sys.path.append('../network/')
sys.path.append('../model/')

import os
import glob
import time
import torch
import logging
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from utils import *
from main_loading import *
from main_network import *
from main_model_rec import *
from main_model_hsc import *
from main_model_abc import *
from main_model_one_class import *
from sklearn.metrics import roc_auc_score

# Must specify: random_state_train, n, mix, ra, net, op

# Initialize the parser
parser = argparse.ArgumentParser()
# >>>> set different random_states
parser.add_argument('-rst', '--random_state_train', type=int, default=42)

# Arguments for main_loading
parser.add_argument('-ln', '--loader_name', type=str, default='gaussian9d_hard')
parser.add_argument('--n', type=int, default=2000)
parser.add_argument('--mix', type=int, default=1)
parser.add_argument('-ra', '--ratio_abnormal', type=float, default=0.1)
parser.add_argument('--n_features', type=int, default=9)


# Arguments for main_network
parser.add_argument('-nt', '--net_name', type=str, default='gaussian9d_rec',
                    help='[Choice]: gaussian9d_one_class, gaussian9d_rec')

# Arguments for main_model
parser.add_argument('-pt', '--pretrain', type=int, default=1,
                    help='[Choice]: Only apply to DeepSAD model: 1 if True, 0 if False')
parser.add_argument('--load_model', type=str, default='',
                    help='[Example]: ./model.tar')
parser.add_argument('-op', '--optimizer_', type=str, default='rec_unsupervised',
                    help='[Choice]: one_class, one_class_unsupervised, rec, rec_unsupervised')
parser.add_argument('-et', '--eta_str', default=100,
                    help='The _% representation of eta - choose from 100, 50, 25, etc.')
parser.add_argument('--optimizer_name', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--ae_lr', type=float, default=0.001)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--ae_n_epochs', type=int, default=100)
parser.add_argument('--lr_milestones', type=int, default='66')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--weight_decay', type=float, default=0.5e-6)
parser.add_argument('--ae_weight_decay', type=float, default=0.5e-3)
parser.add_argument('-gpu', '--device_no', type=int, default=1)
parser.add_argument('--n_jobs_dataloader', type=int, default=0)
parser.add_argument('--save_ae', type=int, default=1,
                    help='Only apply to Deep SAD model.')
parser.add_argument('--load_ae', type=int, default=0,
                    help='Only apply to Deep SAD model.')

p = parser.parse_args()

# ===========================================
# 0.1. Parameters
# ===========================================
# Exract from parser
print('Loading parameters...')
random_state_train, loader_name, n, mix = p.random_state_train, p.loader_name, p.n, bool(p.mix)
ratio_abnormal, n_features = p.ratio_abnormal, p.n_features
net_name, pretrain, load_model = p.net_name, int(p.pretrain), p.load_model
optimizer_, eta_str, optimizer_name = p.optimizer_, p.eta_str, p.optimizer_name
ae_lr, lr, n_epochs, ae_n_epochs, batch_size = p.ae_lr, p.lr, p.n_epochs, p.ae_n_epochs, p.batch_size
weight_decay, ae_weight_decay, device_no, n_jobs_dataloader = p.weight_decay, p.ae_weight_decay, p.device_no, p.n_jobs_dataloader
save_ae, load_ae = bool(p.save_ae), bool(p.load_ae)
lr_milestones = p.lr_milestones

# Define addional parameters
lr_milestones = tuple(i for i in range(lr_milestones, n_epochs, lr_milestones))
torch.manual_seed(random_state_train)
device = 'cuda:{}'.format(device_no)
eta = float(eta_str * 0.01)
label_normal = (0,)

# Debug
print('mix', mix)
print('net_name', net_name)
print('optimizer_', optimizer_)

# ===========================================
# 0.2. Define Path
# ===========================================
# Define folder to save the model and relating results
# Note that we delete pretrain here; the default setting is pretrain.
out_path = f'../result/{loader_name}/models'

print('Checking paths...')
if optimizer_ in ['rec', 'one_class', 'hsc', 'abc']:
    folder_name = f'[model]_{optimizer_}_[n]_{n}_[ra]_{ratio_abnormal}_[lr]_{lr}_[epoch]_{n_epochs}_[net]_{net_name}'
elif optimizer_ in ['rec_unsupervised', 'one_class_unsupervised']:
    folder_name = f'[model]_{optimizer_}_[n]_{n}_[lr]_{lr}_[epoch]_{n_epochs}_[net]_{net_name}'

final_path = Path(out_path) / folder_name
if not os.path.exists(out_path): os.makedirs(out_path)
if not os.path.exists(final_path): os.makedirs(final_path)

# Define the path for others
log_path = Path(final_path) / 'training.log'
model_path = Path(final_path) / 'model.tar'
results_path = Path(final_path) / 'results.json'
ae_results_path = Path(final_path) / 'ae_results.json'

# ===========================================
# 0.3. Setup Logger
# ===========================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(log_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
print(final_path)


# ############################################
# 1. Model Training
# ############################################
# Initialize data
dataset = load_dataset(loader_name=loader_name,
                       random_state=random_state_train,
                       n=n,
                       mix=mix,
                       ratio_abnormal=ratio_abnormal,
                       n_features=n_features)

# Load Deep SAD model
if optimizer_ in ['one_class', 'one_class_unsupervised']:
    # Define model
    model = OneClassModel(optimizer_, eta)
    model.set_network(net_name)

    # Load other models if specified
    if load_model:
        logger.info('Loading model from {}'.format(load_model))
        model.load_model(model_path=load_model,
                         load_ae=True,
                         map_location=device)
    # Pretrain if specified
    if pretrain:
        logger.info('I am pre-training for you.')
        model.pretrain(dataset, optimizer_name, ae_lr, ae_n_epochs, lr_milestones,
                       batch_size, ae_weight_decay, device, n_jobs_dataloader)
        model.save_ae_results(export_json=ae_results_path)

# Load Reconstruction model
elif optimizer_ in ['rec', 'rec_unsupervised']:
    model = RecModel(optimizer_, eta)
    model.set_network(net_name)

elif optimizer_ == 'hsc':
    model = HSCModel()
    model.set_network(net_name)

elif optimizer_ == 'abc':
    model = ABCModel()
    model.set_network(net_name)

# Training model
model.train(dataset, eta, optimizer_name, lr, n_epochs, lr_milestones,
            batch_size, weight_decay, device, n_jobs_dataloader, label_normal, False)


# ############################################
# 2. Model Saving
# ############################################
# Test and Save model
model.test(dataset, device, n_jobs_dataloader, label_normal)
model.save_results(export_json=results_path)
model.save_model(export_model=model_path, save_ae=save_ae)

print('Finished. Now I am going to bed. Bye.')
