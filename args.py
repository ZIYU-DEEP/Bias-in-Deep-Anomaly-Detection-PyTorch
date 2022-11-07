"""
[Title] arguments.py
[Usage] The file to feed in arguments.
"""

import argparse

# Initialize the parser
parser = argparse.ArgumentParser()

# Arguments for config FashionMNIST
parser.add_argument('-cfg', '--config', type=int, default=0,
                    help='If config, set to be 1; else 0')
parser.add_argument('-lb_a_l', '--label_abnormal_list',
                    action='append', default=[1])
parser.add_argument('-ra_a', '--ratio_a', type=float, default=0.1,
                    help='The ratio of the first element in training')
parser.add_argument('-ra_b', '--ratio_b', type=float, default=0.9,
                    help='The ratio of the second element in training')

# Arguments for main_loading
parser.add_argument('-ln', '--loader_name', type=str, default='fmnist',
                    help='The name for the dataset to be loaded.')
parser.add_argument('-rt', '--root', type=str, default='./data/',
                    help='The root for the data folder.')
parser.add_argument('-fn', '--filename', type=str, default='FashionMNIST',
                    help='The filename for your data, e.g., MNIST.')
parser.add_argument('-rr', '--results_root', type=str, default='./results',
                    help='The directory to save results.')
parser.add_argument('-lb_n', '--label_normal', type=int, default=0,
                    help='The normal data needed in training the model.')
parser.add_argument('-lb_a', '--label_abnormal', type=int, default=1,
                    help='The abnormal data needed in training the model.')
parser.add_argument('-ra', '--ratio_abnormal', type=float, default=0.1,
                    help='The amount of abnormal data needed in training.')
parser.add_argument('-l', '--test_list', type=str, default='1',
                    help='The label list to test, e.g. 1, 1-2-3, etc.')

# Arguments for main_network
parser.add_argument('-nt', '--net_name', type=str, default='fmnist_LeNet_one_class',
                    help='[Choice]: synthetic_one_class, synthetic_rec')
parser.add_argument('-rp', '--rep_dim', type=int, default=128,
                    help='The hidden dimensions for the latent vector.')
parser.add_argument('-xd', '--x_dim', type=int, default=8,
                    help='The input dimension for x.')
parser.add_argument('-hd', '--h_dims', type=str, default='32-16',
                    help='The hidden dimensions for the MLP network.')
parser.add_argument('-b', '--bias', action='store_true',
                    help='Include this parameter if using bias.')

# Arguments for main_model
parser.add_argument('-op', '--optimizer_', type=str, default='one_class',
                    help='The anomaly detection model used for optimizer.',
                    choices=['one_class', 'one_class_unsupervised',
                            'rec', 'rec_unsupervised',
                            'abc', 'hsc'])
parser.add_argument('-pt', '--pretrain', type=int, default=1,
                    help='[Choice]: Only apply to DeepSAD model: 1 if True, 0 if False')
parser.add_argument('-mdl', '--load_model', type=str, default='',
                    help='[Example]: ./model.tar')
parser.add_argument('-et', '--eta_str', default=100,
                    help='The _% representation of eta - choose from 100, 50, 25, etc.')
parser.add_argument('-opn', '--optimizer_name', type=str, default='adam')
parser.add_argument('-lr', '--lr', type=float, default=0.001)
parser.add_argument('-ae_lr', '--ae_lr', type=float, default=0.001)
parser.add_argument('-ne', '--n_epochs', type=int, default=200)
parser.add_argument('-ane', '--ae_n_epochs', type=int, default=100)
parser.add_argument('-lm', '--lr_milestones', type=int, default='80')
parser.add_argument('-bs', '--batch_size', type=int, default=128)
parser.add_argument('-wd', '--weight_decay', type=float, default=0.5e-6)
parser.add_argument('-awd', '--ae_weight_decay', type=float, default=0.5e-3)
parser.add_argument('-device', '--device', type=str, default='cuda',
                    help='Use cpu, cuda, or cuda:1, etc.')
parser.add_argument('-nj', '--n_jobs_dataloader', type=int, default=0)
parser.add_argument('-sa', '--save_ae', type=int, default=1,
                    help='Only apply to Deep SAD model.')
parser.add_argument('-la', '--load_ae', type=int, default=0,
                    help='Only apply to Deep SAD model.')

p = parser.parse_args()
