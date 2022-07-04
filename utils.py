import pandas as pd
import argparse
import os
import json
import numpy as np
import logging
from PIL import Image
import boto3

import glob
import cv2

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import albumentations as A
import gc

from tqdm import tqdm
from torch.utils.data import Dataset
from albumentations.pytorch import transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

config = {
    "model_name": 'convnext_base_384_in22ft1k',
    "sch": 'CosineAnnealingLR',
}


class EMB_Dataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.img_dir = df['PATH'].values
        self.labels = df['new_labels'].values
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_dir[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels[index]
        if self.transform:
            img = self.transform(image=img)["image"]

        return {'image': img,
                'new_labels': torch.tensor(label, dtype=torch.long),
                'path': img_path}


def data_transforms_img(img_size):
    data_transforms = {
        "train": A.Compose([
            # 이미지의 maxsize를 max_size로 rescale합니다. aspect ratio는 유지.
            A.LongestMaxSize(max_size=int(img_size * 1.0)),
            # min_size보다 작으면 pad
            A.PadIfNeeded(min_height=int(img_size * 1.0), min_width=int(img_size * 1.0),
                          border_mode=cv2.BORDER_CONSTANT),
            # A.ToGray(p=1),
            A.OneOf([
                A.MotionBlur(p=1),
                A.OpticalDistortion(p=1),
                A.GaussNoise(p=1),
                A.CoarseDropout(p=1, max_holes=8, max_height=8, max_width=8, min_holes=8, min_height=8, min_width=8),
                A.ISONoise(always_apply=False, p=1.0, intensity=(0.1, 0.5), color_shift=(0.01, 0.05))
            ], p=1),
            A.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            A.CLAHE(always_apply=False, p=0.5, clip_limit=(1, 5), tile_grid_size=(8, 8)),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            transforms.ToTensorV2()]),

        "valid": A.Compose([
            # 이미지의 maxsize를 max_size로 rescale합니다. aspect ratio는 유지.
            A.LongestMaxSize(max_size=int(img_size * 1.0)),
            # min_size보다 작으면 pad
            A.PadIfNeeded(min_height=int(img_size * 1.0), min_width=int(img_size * 1.0),
                          border_mode=cv2.BORDER_CONSTANT),
            # A.ToGray(p=1),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            transforms.ToTensorV2()])
    }
    return data_transforms


def criterion(outputs, labels):
    # return nn.CrossEntropyLoss()(outputs, labels)
    return nn.CrossEntropyLoss()(outputs, labels)


############# train one epoch ###################################
def training_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:

        img = data['image'].to(device, dtype=torch.float)
        labels = data['new_labels'].to(device, dtype=torch.long)

        batch_size = img.size(0)

        outputs = model(img)
        loss = criterion(outputs, labels)
        loss.backward()

        if (step + 1) % 1 == 0:
            optimizer.step()

            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    return epoch_loss


@torch.inference_mode()
def valid_epoch(model, dataloader, device, epoch):
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['new_labels'].to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

    gc.collect()

    return epoch_loss


def make_csv_file(path):
    labels = []
    img_path = []
    logger.info("make_csv_file")
    for label in os.listdir(path):
        img_path.extend(glob.glob(os.path.join(path, label) + '/*.jpg'))
        labels.extend([label] * len(glob.glob(os.path.join(path, label) + '/*.jpg')))
        img_path.extend(glob.glob(os.path.join(path, label) + '/*.JPG'))
        labels.extend([label] * len(glob.glob(os.path.join(path, label) + '/*.JPG')))
    csv = pd.DataFrame({'PATH': img_path,
                        'labels': labels})
    return csv


def test_label_encoding(df, encoder):
    df['new_labels'] = 'None'
    logger.info("test_label_encoding")
    for i in range(len(df)):
        if df['labels'][i] in encoder.classes_:
            df['new_labels'][i] = int((np.where(encoder.classes_ == df['labels'][i])[0]))
            df[df['new_labels'] != 'None']
    return df[df['new_labels'] != 'None']


#    AWS dataset
class aws_EMB_Dataset(Dataset):
    def __init__(self, df, fs, transforms=None):
        self.fs = fs
        self.df = df
        self.img_dir = df['PATH'].values
        self.labels = df['new_labels'].values
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_dir[index]
        with self.fs.open(img_path) as f:
            pil_src = Image.open(f)
            numpy_src = np.array(pil_src)
            img = cv2.cvtColor(numpy_src, cv2.COLOR_RGB2BGR)
            label = self.labels[index]

        if self.transform:
            img = self.transform(image=img)["image"]

        return {'image': img,
                'new_labels': torch.tensor(label, dtype=torch.long),
                'path': img_path}


###############   make csv in aws ################
def aws_make_csv_file(path, conn, bucket, subfolder):
    s3 = boto3.resource('s3')
    my_bucket = s3.Bucket(bucket)
    labels = []
    img_path = []

    for label in my_bucket.objects.filter(Prefix=subfolder + path):
        img_path.extend([f's3://' + bucket + '/' + label.key])
        labels.extend([label.key[len(subfolder + path):].split('/')[0]])

    csv = pd.DataFrame({'PATH': img_path, 'labels': labels})

    return csv
