#!/bin/bash
cd ..

# Arguments for loading
root=./data/
loader_name=fmnist
filename=FashionMNIST
results_root=./results
label_normal=2
label_abnormal=4
ratio_abnormal=0.1
test_list=1-3-5-6-7

# Arguments for model; choose from the following:
# [Unsupervised model] one_class_unsupervised, rec_unsupervised
# [Supervised model] one_class, rec, hsc, abc
optimizer_=hsc

# Arguments for model and optimizer
optimizer_=hsc
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
python main.py -ln ${loader_name} -rt ${root} -rr ${results_root} -fn ${filename} -lb_n ${label_normal} -lb_a ${label_abnormal} -ra ${ratio_abnormal} -nt ${net_name} -op ${optimizer_} -l ${test_list} -ne ${n_epochs} -ane ${ae_n_epochs} -device ${device} -lr ${lr} -ae_lr ${ae_lr} -lm ${lr_milestones} -bs ${batch_size} -wd ${weight_decay} -awd ${ae_weight_decay}
