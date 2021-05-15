"""
[Title] main_pac_gaussian_test.py
[Description] The main file to test the synthetic dataset.
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
import random
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
from sklearn.metrics import roc_auc_score

# Must specify: n_eval
# When test on dataset with different n_eval, set the the ra_train to be the same
# in order to keep the model / scoring function the same

# Initialize the parser
parser = argparse.ArgumentParser()
# >>>> set different random_states for val and test
parser.add_argument('-rst', '--random_state_test', type=int, default=666)

# Arguments for main_loading
parser.add_argument('-ln', '--loader_name', type=str, default='gaussian9d')
parser.add_argument('--n', type=int, default=2000)
parser.add_argument('-n_t', '--n_test', type=int, default=20000)
parser.add_argument('-n_e', '--n_eval', type=int, default=100)
parser.add_argument('--mix', type=int, default=1)
parser.add_argument('-ra_train', '--ratio_abnormal_train', type=float, default=0.01)
parser.add_argument('-ra_eval', '--ratio_abnormal_eval', type=float, default=0.01)
parser.add_argument('-ra_test', '--ratio_abnormal_test', type=float, default=0.01)
parser.add_argument('--n_features', type=int, default=9)


# Arguments for main_network
parser.add_argument('-nt', '--net_name', type=str, default='gaussian9d_one_class',
                    help='[Choice]: gaussian9d_one_class, gaussian9d_rec')

# Arguments for main_model
parser.add_argument('--load_model', type=str, default='',
                    help='[Example]: ./model.tar')
parser.add_argument('-op', '--optimizer_', type=str, default='one_class_unsupervised',
                    help='[Choice]: one_class, one_class_unsupervised, rec, rec_unsupervised')
parser.add_argument('-et', '--eta_str', default=100,
                    help='The _% representation of eta - choose from 100, 50, 25, etc.')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--lr_milestones', type=int, default='66')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('-gpu', '--device_no', type=int, default=1)
parser.add_argument('--n_jobs_dataloader', type=int, default=0)


p = parser.parse_args()

# ===========================================
# 0.1. Parameters
# ===========================================
# Exract from parser
print('Loading parameters...')
random_state_test = p.random_state_test
random_state_eval = random.randint(0, np.iinfo(np.int32).max)
loader_name, n, n_eval, n_test, mix = p.loader_name, p.n, p.n_eval, p.n_test, bool(p.mix)
ratio_abnormal_train, ratio_abnormal_eval = p.ratio_abnormal_train, p.ratio_abnormal_eval
ratio_abnormal_test = p.ratio_abnormal_test
n_features, net_name, load_model = p.n_features, p.net_name, p.load_model
optimizer_, eta_str= p.optimizer_, p.eta_str
lr, n_epochs, batch_size = p.lr, p.n_epochs, p.batch_size
device_no, n_jobs_dataloader = p.device_no, p.n_jobs_dataloader
lr_milestones = p.lr_milestones

# Define addional parameters
lr_milestones = tuple(i for i in range(lr_milestones, n_epochs, lr_milestones))
torch.manual_seed(random_state_test)
device = 'cuda:{}'.format(device_no)
eta = float(eta_str * 0.01)
label_normal = (0,)
label_abnormal = (1,)

# ===========================================
# 0.2. Define Path
# ===========================================
# Get path for models
print('Checking paths...')

out_path = f'../result/{loader_name}/models'
out_path_test = f'../result/{loader_name}/reports'

# Get the folder name for the model
if optimizer_ in ['rec', 'one_class']:
    folder_name = f'[model]_{optimizer_}_[n]_{n}_[ra]_{ratio_abnormal_train}_[lr]_{lr}_[epoch]_{n_epochs}_[net]_{net_name}'
elif optimizer_ in ['rec_unsupervised', 'one_class_unsupervised']:
    folder_name = f'[model]_{optimizer_}_[n]_{n}_[lr]_{lr}_[epoch]_{n_epochs}_[net]_{net_name}'

# Get the folder name to save the data
folder_name_test = f'[ra_test]_{ratio_abnormal_test}_[ra_eval]_{ratio_abnormal_eval}_[n_eval]_{n_eval}_{folder_name}'
print(f'Destination: {folder_name_test}!')

final_path = Path(out_path) / folder_name
final_path_test = Path(out_path_test) / folder_name_test

# Define the path for model
log_path = Path(final_path) / 'training.log'
model_path = Path(final_path) / 'model.tar'

# Define the path for test
result_txt_path = Path(final_path_test) / 'result.txt'

# Check path
if not os.path.exists(out_path_test): os.makedirs(out_path_test)
if not os.path.exists(final_path_test): os.makedirs(final_path_test)


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
# 1. Data Loading
# ############################################
# Initialize data
dataset_eval = load_dataset(loader_name=loader_name,
                            random_state=random_state_eval,
                            n=n_eval,
                            mix=True,
                            ratio_abnormal=ratio_abnormal_eval,
                            n_features=n_features)


dataset_test = load_dataset(loader_name=loader_name,
                            random_state=random_state_test,
                            n=n_test,
                            mix=True,
                            ratio_abnormal=ratio_abnormal_test,
                            n_features=n_features)


# ############################################
# 2. Model Loading
# ############################################
print('Loading model from {}'.format(model_path))

if optimizer_ in ['one_class', 'one_class_unsupervised']: model = OneClassModelEval(optimizer_, eta)
elif optimizer_ in ['rec', 'rec_unsupervised']: model = RecModelEval(optimizer_, eta)

model.set_network(net_name)
model.load_model(model_path=model_path, map_location=device)


# ############################################
# 2. Model Evaluating
# ############################################
model.test(dataset=dataset_eval,
           eta=eta,
           batch_size=batch_size,
           device=device,
           n_jobs_dataloader=n_jobs_dataloader,
           label_normal=label_normal)

indices_, labels_, scores_ = zip(*model.results['test_scores'])
indices_, labels_, scores_ = np.array(indices_), np.array(labels_), np.array(scores_)

result_df = pd.DataFrame()
result_df['indices'], result_df['labels'], result_df['scores'] = indices_, labels_, scores_

result_df.drop('indices', inplace=True, axis=1)
df_normal = result_df[result_df.labels.isin(label_normal)]
df_abnormal = result_df[result_df.labels.isin(label_abnormal)]

# Save the threshold
cut_95 = df_normal.scores.quantile(0.95)
y_95 = [1 if e > cut_95 else 0 for e in df_abnormal['scores'].values]
recall_95 = sum(y_95) / len(y_95)


# ############################################
# 3. Model Testing
# ############################################
# Note this is differnt from training model's model.test
model.test(dataset=dataset_test,
           eta=eta,
           batch_size=batch_size,
           device=device,
           n_jobs_dataloader=n_jobs_dataloader,
           label_normal=label_normal)

indices_t, labels_t, scores_t = zip(*model.results['test_scores'])
indices_t, labels_t, scores_t = np.array(indices_t), np.array(labels_t), np.array(scores_t)

result_df_t = pd.DataFrame()
result_df_t['indices'], result_df_t['labels'], result_df_t['scores'] = indices_t, labels_t, scores_t

result_df_t.drop('indices', inplace=True, axis=1)
df_normal_t = result_df_t[result_df_t.labels.isin(label_normal)]
df_abnormal_t = result_df_t[result_df_t.labels.isin(label_abnormal)]

# Get the results for recall
y_95_t = [1 if e > cut_95 else 0 for e in df_abnormal_t['scores'].values]
recall_95_t = sum(y_95_t) / len(y_95_t)

# Get the results for fpr
y_95_t_fpr = [1 if e > cut_95 else 0 for e in df_normal_t['scores'].values]
fpr_95_t = sum(y_95_t_fpr) / len(y_95_t_fpr)

print(f'FPR and Recall on evaluation dataset: 0.05; {recall_95}')
print(f'FPR and Recall on test dataset: {fpr_95_t}; {recall_95_t}')

# ############################################
# 4. Save Results
# ############################################
f = open(result_txt_path, 'a')
f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
f.write(f'\n[n_eval] {n_eval} [n_test] {n_test}')
f.write(f'\n[cut_95_eval] {cut_95}')
f.write(f'\n[recall_eval] {recall_95} [recall_test] {recall_95_t}')
f.write(f'\n[fpr_eval] {0.05} [fpr_test] {fpr_95_t}\n\n')
f.close()

print('Finished. Now I am going to bed. Bye.')
