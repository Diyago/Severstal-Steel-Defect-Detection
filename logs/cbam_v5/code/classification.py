# Pre-trained weights: https://github.com/facebookresearch/WSL-Images
# Apex: https://github.com/NVIDIA/apex
# Borrowed a lot from abhishek, so give him an upvote: https://www.kaggle.com/abhishek/very-simple-pytorch-training-0-59

# Parameters

lr = 2e-5
img_size = 224
batch_size = 32
n_epochs = 10
n_freeze = 1
classes = (0, 1, 2, 3)
coef = [0.5, 1.5, 2.5, 3.5]


# Libraries

import torch
import os
import gc
import sys
import cv2
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from albumentations import Compose, RandomBrightnessContrast, ShiftScaleRotate
from albumentations.pytorch import ToTensor
from torch.utils.data import Dataset
from torchvision.models.resnet import ResNet, Bottleneck
import torch.optim as optim

# Install Apex for mixed precision

print('Starting Apex installation ...')

FNULL = open(os.devnull, 'w')
process = subprocess.Popen(
    'pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../input/nvidia-apex/apex/apex',
    shell=True,
    stdout=FNULL, stderr=subprocess.STDOUT)
process.wait()

if process.returncode==0:
    print('Apex successfully installed')

from apex import amp

# Functions


class RetinopathyDatasetTrain(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptospreprocessed224x/train_images_processed_224x/train_images_processed_224x/', self.data.loc[idx, 'id_code'] + '.png.png') # typo
        im = cv2.imread(img_name)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        if self.transform:
            augmented = self.transform(image=im)
            im = augmented['image']
        return {'image': im, 'labels': label}

class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data.loc[idx, 'id_code'] + '.png')
        im = cv2.imread(img_name)
        im = circle_crop(im)
        im = cv2.resize(im, (img_size, img_size))
        if self.transform:
            augmented = self.transform(image=im)
            im = augmented['image']
        return {'image': im}

def _resnext(path, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    model.load_state_dict(torch.load(path))
    return model

def resnext101_32x16d_wsl(path, progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x16 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_
    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 16
    return _resnext(path, Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)

def train_model(model,  n_epochs, classification=True):

    for epoch in range(n_epochs):
        if epoch == n_freeze:
            for param in model.parameters():
                param.requires_grad = True
        tr_loss = 0
        counter = 0
        model.train()
        print('Epoch {}/{}'.format(epoch, n_epochs - 1))
        print('-' * 10)
        for step, batch in enumerate(train_data_loader):
            if classification:
                inputs = batch["image"]
                labels = batch["labels"]
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
            else:
                inputs = batch["image"]
                labels = batch["labels"].view(-1, 1)
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss = tr_loss / len(train_data_loader)
        print('Training Loss: {:.4f}'.format(epoch_loss))

    return model

# processing

transform_train = Compose([
    ShiftScaleRotate(
        shift_limit=0.1,
        scale_limit=0.1,
        rotate_limit=365,
        p=1.0),
    RandomBrightnessContrast(p=1.0),
    ToTensor()
])

transform_test = Compose([
    ToTensor()
])

train_dataset = RetinopathyDatasetTrain(csv_file='../input/aptos2019-blindness-detection/train.csv', transform=transform_train)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

test_dataset = RetinopathyDatasetTest(csv_file='../input/aptos2019-blindness-detection/test.csv', transform=transform_test)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device("cuda:0")

# classification model

model = resnext101_32x16d_wsl(path='../input/ig-resnext101-32x16/ig_resnext101_32x16-c6f796b0.pth')

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(2048, len(classes))
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
plist = [{'params': model.parameters(), 'lr': lr}]
optimizer = optim.Adam(plist, lr=lr)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

model = train_model(model=model,n_epochs=n_epochs,classification=True)

for param in model.parameters():
    param.requires_grad = False

model.eval()

test_preds1 = np.zeros((len(test_dataset), 5))

for i, x_batch in enumerate(tqdm(test_data_loader)):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds1[i * batch_size:(i + 1) * batch_size] = pred.detach().cpu()

test_preds1 = np.argmax(test_preds1, axis=1)

del(model, optimizer, criterion, plist)
gc.collect()
torch.cuda.empty_cache()

# regression model

model = resnext101_32x16d_wsl(path='../input/ig-resnext101-32x16/ig_resnext101_32x16-c6f796b0.pth')

for param in model.parameters():
    param.requires_grad = False

model.fc = torch.nn.Linear(2048, 1)
model.to(device)

criterion = torch.nn.MSELoss()
plist = [{'params': model.parameters(), 'lr': lr}]
optimizer = optim.Adam(plist, lr=lr)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

model = train_model(model=model,n_epochs=n_epochs,classification=False)

for param in model.parameters():
    param.requires_grad = False

model.eval()

test_preds2 = np.zeros((len(test_dataset), 1))

for i, x_batch in enumerate(tqdm(test_data_loader)):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds2[i * batch_size:(i + 1) * batch_size] = pred.detach().cpu()

i = 0

for pred in test_preds2:
    if pred < coef[0]:
        test_preds2[i] = 0
        i += 1
    elif pred >= coef[0] and pred < coef[1]:
        test_preds2[i] = 1
        i += 1
    elif pred >= coef[1] and pred < coef[2]:
        test_preds2[i] = 2
        i += 1
    elif pred >= coef[2] and pred < coef[3]:
        test_preds2[i] = 3
        i += 1
    else:
        test_preds2[i] = 4
        i += 1

test_preds2 = test_preds2.reshape(-1,)

# combine

final_pred = np.round((test_preds1+test_preds2)/2)

# submit

sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample.diagnosis = final_pred.astype(int)
sample.to_csv("submission.csv", index=False)