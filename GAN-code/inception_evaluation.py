# Copyright (c) ElementAI and its affiliates.
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Written by Amjad Almahairi (amjad.almahairi@elementai.com)

import torch
from torch import nn

from torch.nn import functional as F
import torch.utils.data
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms

import numpy as np
from scipy.stats import entropy
import os


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32*5*5, 128)
        self.fc2 = nn.Linear(128, 96)
        self.fc3 = nn.Linear(96, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, p=0.25, training=self.training)
        out = F.relu(self.fc2(out))
        out = F.dropout(out, p=0.25, training=self.training)
        out = self.fc3(out)
        return out

    
def mnist_inception_score(imgs, device=None, batch_size=500, splits=1, model_path='mnist_classifier.pt'):
    """Computes the inception score of the generated images imgs
    adapted from https://github.com/sbarratt/inception-score-pytorch/blob/master/inception_score.py

    imgs -- Torch tensor (bsx1x32x32) of images normalized in the range [-1, 1]
    device -- gpu/cpu
    batch_size -- batch size for feeding into mnist_classifier
    splits -- number of splits
    model_path -- path to pretrained mnist classifier
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Load inception model
    mnist_classifier = LeNet().to(device)
    mnist_classifier.load_state_dict(torch.load(model_path))
    mnist_classifier.eval()

    # Get predictions
    preds = np.zeros((N, 10))

    with torch.no_grad():
        i = 0
        total_i = 0
        while total_i < N:
            batch = imgs[i*batch_size:(i+1)*batch_size]
            batch_size_i = batch.size()[0]
            batch_preds = mnist_classifier(batch)
            batch_preds = F.softmax(batch_preds, dim=1).cpu().numpy()
            preds[i*batch_size:i*batch_size + batch_size_i] = batch_preds
            i += 1
            total_i += batch_size_i

    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
