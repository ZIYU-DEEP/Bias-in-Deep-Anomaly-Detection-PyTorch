import torch
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.utils import shuffle

print('...')
satimage = fetch_openml(name='satimage')
X = satimage.data.astype('float32')
y = satimage.target.astype('float').astype('int32')
X, y = shuffle(X, y, random_state=42)
data_  = (X, y)
torch.save(data_, '../data/satimage.pkl')

print('...')
phish_url = fetch_openml(name='phish_url', version=1)
target_to_y = {'Defacement':0, 'benign':1, 'malware':2, 'phishing':3, 'spam':4}
X = phish_url.data.astype('float32')
y = np.array([target_to_y[i] for i in phish_url.target]).astype('int32')
X, y = shuffle(X, y, random_state=42)
data  = (X, y)
torch.save(data, '../data/phish_url.pkl')

print('...')
covertype = fetch_openml(name='covertype', version=3)
X = covertype.data.astype('float32')
y = covertype.target.astype('float32').astype('int32')
X, y = shuffle(X, y, random_state=42)
data  = (X, y)
torch.save(data, '../data/covertype.pkl')

print('...')
shuttle = fetch_openml(name='shuttle')
X = shuttle.data.astype('float32')
y = shuttle.target.astype('float32').astype('int32')
X, y = shuffle(X, y, random_state=42)
data  = (X, y)
torch.save(data, '../data/shuttle.pkl')

print('...')
gas_drift = fetch_openml(name='gas-drift')
X = gas_drift.data.astype('float32')
y = gas_drift.target.astype('float32').astype('int32')
X, y = shuffle(X, y, random_state=42)
data  = (X, y)
torch.save(data, '../data/gas_drift.pkl')

print('...')
kmnist_49 = fetch_openml(name='Kuzushiji-49')
X = kmnist_49.data.astype('float32')
y = kmnist_49.target.astype('float32').astype('int32')
X, y = shuffle(X, y, random_state=42)
data  = (X.reshape(-1, 28, 28), y)
torch.save(data, '../data/kmnist_49.pkl')

print('...')
usps = fetch_openml(name='usps', version=2)
X = usps.data.astype('float32')
y = usps.target.astype('float32').astype('int32')
X, y = shuffle(X, y, random_state=42)
data  = (X.reshape(-1, 16, 16), y)
torch.save(data, '../data/usps.pkl')

print('...')
SVHN = fetch_openml(name='SVHN')
X = SVHN.data.astype('float32')
y = SVHN.target.astype('float32').astype('int32')
X, y = shuffle(X, y, random_state=42)
data  = (X, y)
torch.save(data, '../data/SVHN.pkl')
