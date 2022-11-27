#!/bin/bash
cd ..

# Arguments for loading
root=./data/
loader_name=real
filename=satimage
results_root=./results
label_normal=3
label_abnormal=1
ratio_abnormal=0.1
test_list=1-2-4-5-7

# Arguments for network
net_name=satimage_one_class         # net_name=satimage_hsc
rep_dim=8
x_dim=36
h_dims=32-16

# Arguments for model; choose from the following:
# [Unsupervised model] one_class_unsupervised, rec_unsupervised
# [Supervised model] one_class, rec, hsc, abc
optimizer_=one_class               # optimizer_=hsc
pretrain=1                         # pretrain=0

# Arguments for the optimizer
lr=0.001
ae_lr=0.001
n_epochs=200
ae_n_epochs=100
lr_milestones=80
batch_size=128
weight_decay=0.5e-6
ae_weight_decay=0.5e-3

# Other unimportant arguments
device=cpu

# Run
python3 main.py -ln ${loader_name} -rt ${root} -rr ${results_root} -fn ${filename} -lb_n ${label_normal} -lb_a ${label_abnormal} -ra ${ratio_abnormal} -nt ${net_name} -op ${optimizer_} -l ${test_list} -ne ${n_epochs} -ane ${ae_n_epochs} -device ${device} -lr ${lr} -ae_lr ${ae_lr} -lm ${lr_milestones} -bs ${batch_size} -wd ${weight_decay} -awd ${ae_weight_decay} -rp ${rep_dim} -xd ${x_dim} -hd ${h_dims} -pt ${pretrain}
