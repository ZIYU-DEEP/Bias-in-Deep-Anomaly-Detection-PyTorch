"""
[Title] main_vision.py
[Description] The main file to run the models for torch.vision datasets.
"""

# ############################################
# 0. Preparation
# ############################################
import sys
sys.path.append('../dataset/')
sys.path.append('../network/')
sys.path.append('../model/')

import os
import glob
import time
import torch
import joblib
import logging
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from utils import *
from main_loading import *
from main_network import *
from main_model_rec import *
from main_model_one_class import *
from main_model_hsc import *
from main_model_abc import *
from sklearn.metrics import roc_auc_score

# Initialize the parser
parser = argparse.ArgumentParser()

# Arguments for config FashionMNIST
parser.add_argument('-cfg', '--config', type=int, default=0,
                    help='If config, set to be 1; else 0')
parser.add_argument('-lb_a_l', '--label_abnormal_list',
                    action='append', default=[6, 7])
parser.add_argument('-ra_a', '--ratio_a', type=float, default=0.1,
                    help='The ratio of the first element in training')
parser.add_argument('-ra_b', '--ratio_b', type=float, default=0.9,
                    help='The ratio of the second element in training')


# Arguments for main_loading
parser.add_argument('-ln', '--loader_name', type=str, default='fmnist',
                    help='The name for the dataset to be loaded.')
parser.add_argument('-rt', '--root', type=str, default='/net/leksai/data/',
                    help='The root for the data folder.')
parser.add_argument('-lb_n', '--label_normal', type=int, default=2,
                    help='The normal data needed in training the model.')
parser.add_argument('-lb_a', '--label_abnormal', type=int, default=4,
                    help='The abnormal data needed in training the model.')
parser.add_argument('-ra', '--ratio_abnormal', type=float, default=0.1,
                    help='The amount of abnormal data needed in training.')
parser.add_argument('-l', '--test_list', action='append', default=[3, 4, 50, 49, 52, 53, 527, 620, 429, 430],
                    help='For satimage: -l 4 -l 1 -l 2 -l 5 -l 7')

# Arguments for main_network
parser.add_argument('-nt', '--net_name', type=str, default='fmnist_LeNet_one_class',
                    help='[Choice]: synthetic_one_class, synthetic_rec')

# Arguments for main_model
parser.add_argument('-op', '--optimizer_', type=str, default='one_class',
                    help='[Choice]: one_class, one_class_unsupervised, rec, rec_unsupervised')
parser.add_argument('-pt', '--pretrain', type=int, default=1,
                    help='[Choice]: Only apply to DeepSAD model: 1 if True, 0 if False')
parser.add_argument('--load_model', type=str, default='',
                    help='[Example]: ./model.tar')
parser.add_argument('-et', '--eta_str', default=100,
                    help='The _% representation of eta - choose from 100, 50, 25, etc.')
parser.add_argument('--optimizer_name', type=str, default='adam')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--ae_lr', type=float, default=0.001)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--ae_n_epochs', type=int, default=100)
parser.add_argument('--lr_milestones', type=int, default='80')
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
ratio_a, ratio_b = p.ratio_a, p.ratio_b
config, label_abnormal_list = p.config, p.label_abnormal_list
loader_name, root = p.loader_name, p.root
label_normal, label_abnormal = tuple([p.label_normal]), tuple([p.label_abnormal])
if config == 1:
    label_abnormal = tuple(label_abnormal_list)
ratio_abnormal, test_list = p.ratio_abnormal, p.test_list

net_name, pretrain, load_model = p.net_name, int(p.pretrain), p.load_model
optimizer_, eta_str, optimizer_name = p.optimizer_, p.eta_str, p.optimizer_name
ae_lr, lr, n_epochs, ae_n_epochs, batch_size = p.ae_lr, p.lr, p.n_epochs, p.ae_n_epochs, p.batch_size
weight_decay, ae_weight_decay, device_no, n_jobs_dataloader = p.weight_decay, p.ae_weight_decay, p.device_no, p.n_jobs_dataloader
save_ae, load_ae = bool(p.save_ae), bool(p.load_ae)
lr_milestones = tuple(i for i in range(p.lr_milestones, n_epochs, p.lr_milestones))

# Define addional parameters
device = 'cuda:{}'.format(device_no)
eta = float(eta_str * 0.01)

# Debug
print('net_name', net_name)
print('optimizer_', optimizer_)

# ===========================================
# 0.2. Define Path
# ===========================================
# Define folder to save the model and relating results
# Note that we delete pretrain here; the default setting is pretrain.
out_path = f'../result/{loader_name}'

print('Checking paths...')
if optimizer_ in ['rec', 'one_class', 'hsc', 'abc']:
    load_method = 2
    if config == 0:
        folder_name = f'[semi-model]_{optimizer_}_[lb_n]_{p.label_normal}_[lb_a]_{p.label_abnormal}_[ra]_{ratio_abnormal}_[lr]_{lr}_[epoch]_{n_epochs}_[net]_{net_name}'
    elif config == 1:
        folder_name = f'[semi-model]_{optimizer_}_[lb_n]_{p.label_normal}_[lb_a]_{label_abnormal}_[ra_a]_{ratio_a}_[ra_b]_{ratio_b}_[ra]_{ratio_abnormal}_[lr]_{lr}_[epoch]_{n_epochs}_[net]_{net_name}'
elif optimizer_ in ['rec_unsupervised', 'one_class_unsupervised']:
    load_method = 0
    folder_name = f'[un-model]_{optimizer_}_[lb_n]_{p.label_normal}_[lr]_{lr}_[epoch]_{n_epochs}_[net]_{net_name}'

final_path = Path(out_path) / folder_name
if not os.path.exists(out_path): os.makedirs(out_path)
if not os.path.exists(final_path): os.makedirs(final_path)

# Define the path for others
log_path = Path(final_path) / 'training.log'
model_path = Path(final_path) / 'model.tar'
results_path = Path(final_path) / 'results.json'
ae_results_path = Path(final_path) / 'ae_results.json'
cut_results_path = Path(final_path) / 'cut_results.csv'
score_results_path = Path(final_path) / 'score_results.pkl'
recall_results_path = Path(final_path) / 'recall_results.csv'
txt_results_path = Path(final_path) / 'results.txt'
recall_history_path = Path(final_path) / 'recall_history.pkl'

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
                       root=root,
                       train=1,
                       load_method=load_method,
                       label_normal=label_normal,
                       label_abnormal=label_abnormal,
                       ratio_abnormal=ratio_abnormal,
                       ratio_a=ratio_a,
                       ratio_b=ratio_b)

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

# Test and Save model
model.test(dataset, device, n_jobs_dataloader, label_normal)
model.save_results(export_json=results_path)
model.save_model(export_model=model_path, save_ae=save_ae)

print('Finished. Now I am going to bed. Bye.')


# ############################################
# 2. Model Evaluation (Set the Threshold)
# ############################################
# Use model eval to load dataset
if optimizer_ in ['one_class', 'one_class_unsupervised']: model = OneClassModelEval(optimizer_, eta)
elif optimizer_ in ['rec', 'rec_unsupervised']: model = RecModelEval(optimizer_, eta)
elif optimizer_ == 'hsc': model = HSCModelEval()
elif optimizer_ == 'abc': model = ABCModelEval()

model.set_network(net_name)
model.load_model(model_path=model_path, map_location=device)

# Only load normal data, as we just need to set the threshold by FPR
dataset_eval = load_dataset(loader_name=loader_name,
                            root=root,
                            train=0,
                            load_method=0,
                            label_normal=label_normal,
                            threshold_type=1)

# Evaluation on the test part of the dataset used to train
model.test(dataset=dataset_eval,
           eta=eta,
           batch_size=batch_size,
           device=device,
           n_jobs_dataloader=n_jobs_dataloader,
           label_normal=label_normal)

# Get the evaluation results
indices_, labels_, scores_ = zip(*model.results['test_scores'])
indices_, labels_, scores_ = np.array(indices_), np.array(labels_), np.array(scores_)

# Get thresholds
cut_results = {}
for fpr in [0.90, 0.95, 0.99]: # typo here, fpr should be 0.05; here fpr is tnr
    cut_results[fpr] = np.quantile(scores_, fpr)

# Save the thresholds
cut_results_df = pd.DataFrame(cut_results, index=['Cut'])
cut_results_df.to_csv(cut_results_path, sep='\t', index=False)


# ############################################
# 3. Model Testing
# ############################################
# Get started
print('Start testing...')
recall_results = {}
score_results = {}
f = open(txt_results_path, 'a')
f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
f.write(f'\n[folder_name] {folder_name}\n\n')

if loader_name != 'imagenet':
    test_list = range(10)

if loader_name == 'imagenet':
    test_list = [3, 4, 50, 49, 52, 53, 527, 620, 429, 430]

if loader_name == 'dad':
    test_list = [0, 2, 3, 4, 5, 6, 7, 8]

if recall_history_path.is_file():
    recall_history = joblib.load(recall_history_path)
else:
    recall_history = {i: [] for i in test_list}

for label_abnormal_test in test_list:
    # Skip for normal data
    if (label_abnormal_test == p.label_abnormal) and optimizer_ in ['rec', 'one_class', 'abc', 'hsc']:
        trained_type = 1
    else:
        trained_type = 0

    # Print intro
    intro_str = f'[label] {label_abnormal_test}\n'
    print(intro_str); f.write(intro_str)

    # Get the dataset
    dataset_test = load_dataset(loader_name=loader_name,
                                root=root,
                                train=0,
                                load_method=1,
                                label_abnormal=tuple([label_abnormal_test]),
                                ratio_abnormal=ratio_abnormal,
                                ratio_a=ratio_a,
                                ratio_b=ratio_b,
                                trained_type=trained_type)

    # Test on the dataset
    model.test(dataset=dataset_test,
               eta=eta,
               batch_size=batch_size,
               device=device,
               n_jobs_dataloader=n_jobs_dataloader,
               label_normal=label_normal)

    # Get test results
    indices, labels, scores = zip(*model.results['test_scores'])
    indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    score_results[label_abnormal_test] = scores

    # Get the recall
    recall_results_i = {}
    for fpr in [0.90, 0.95, 0.99]:
        # Calculate recall
        y = [1 if e > cut_results[fpr] else 0 for e in scores]
        recall = sum(y) / len(y); recall_results_i[fpr] = recall
        # Print and save
        recall_str = f'[FPR]: {fpr}; [recall]: {recall}\n'
        print(recall_str); f.write(recall_str)
        # Save to history
        if fpr == 0.95:
            recall_history[label_abnormal_test].append(recall)
    recall_results[label_abnormal_test] = recall_results_i

    # Write done mean and std for recall history
    recall_hist_mean = np.mean(recall_history[label_abnormal_test])
    recall_hist_std = np.std(recall_history[label_abnormal_test])
    recall_len = len(recall_history[label_abnormal_test])
    hist_str = f'[mean] {recall_hist_mean}; [std] {recall_hist_std}; [len] {recall_len}.\n'
    print(hist_str); f.write(hist_str)
    f.write('\n')

# Save score results and test results
joblib.dump(score_results, score_results_path)
joblib.dump(recall_history, recall_history_path)
pd.DataFrame(recall_results).to_csv(recall_results_path, sep='\t')

# Finalize
f.write('\n\n'); f.close()
print('Finished. Now I am going to bed. Good luck.')
