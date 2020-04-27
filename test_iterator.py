import collections
from multiprocessing.spawn import freeze_support

import numpy as np
import random
import torchvision
import matplotlib.pyplot as plt
import time
import os
import copy
import math
from pathlib import Path
from tqdm import trange, tqdm
from itertools import islice
from collections import Counter
import torch
from torch.utils import data

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import itertools

# print(torch.cuda.is_available())

LABS_DIR = Path('C:/Labs/')

DATA_DIR = LABS_DIR / 'Data'


# Data augmentation and normalization for training
# for validatin we use normalization and resize (for train we also change the angle and size of the images)

def train_model_simulation(data, model, criterion, optimizer, scheduler, num_epochs=2, checkpoint=None):
    since = time.time()
    print("Starting epochs")
    outer = tqdm(total=num_epochs, desc='Epoch', position=0)
    inner = tqdm(total=500, position=1)

    time_elapsed = time.time() - since
    # load best model weights
    for epoch in range(num_epochs):
        outer.update(1)
        print(f'Starting epoch: {epoch}')
        for phase in ['train', 'val']:
            print(f'Starting phase: {phase}')
            running_loss = 0.0
            running_corrects = 0
            total = 500

            # Handle tqdm inner loop counter
            inner.total = total
            inner.reset()
            inner_total = 0
            for i, (inputs, labels) in enumerate(data[phase]):
                running_corrects = 10
                inner.write(f'input #:{i}, labels={labels} ')  # , end='\r', flush=True
            # if (running_corrects == 0):
            #     print("Skipped inner loop")


# train_model_simulation(data,
#                        None,
#                        None,
#                        None,
#                        None,
#                        num_epochs=3,
#                        checkpoint=None)

if __name__ == '__main__':
    print(torch.__version__)

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.96, 1.0), ratio=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    sample_size = 20
    sample_n = {x: random.sample(list(range(dataset_sizes[x])), sample_size)
                for x in ['train', 'val']}

    image_datasets_reduced = {x: torch.utils.data.Subset(image_datasets[x], sample_n[x])
                              for x in ['train', 'val']}

    # dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
    #                                               shuffle=True, num_workers=4)
    #                for x in ['train', 'val']}

    dataloaders_reduced = {x: torch.utils.data.DataLoader(image_datasets_reduced[x], batch_size=4,
                                                          shuffle=True, num_workers=1)
                           for x in ['train', 'val']}

    # load best model weights

    data = dataloaders_reduced

    # for test in range(5):
    #     for phase in ['train','test']:
    #         cnt = Counter()
    #         for i, (inputs, labels) in enumerate(data['train']):
    #             print(f'input #:{i}, labels={labels} ',  flush=True) #end='\r',
    #             cnt.update(labels.tolist())
    #         print(f'Phase={phase}, counts={cnt}')

    train_model_simulation(data,
                           None, None, None, None,
                           num_epochs=3,
                           checkpoint=None)
