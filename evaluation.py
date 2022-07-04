import heapq
import pathlib
import pickle
import tarfile

import pandas as pd
from tqdm import tqdm

import argparse

import os
import time
import cv2
import gc
import copy
import json
import numpy as np

from collections import defaultdict

from sklearn.preprocessing import LabelEncoder

import albumentations as A
from albumentations.pytorch import transforms

import torch

import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from training import training_epoch, val_epoch
from make_csv import aws_make_csv_file
from Module import EMB_model, EMB_Dataset, aws_EMB_Dataset

import boto3
import s3fs
from sagemaker import get_execution_role
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
config = {
    "model_name": 'convnext_large_384_in22ft1k',
    "sch": 'CosineAnnealingLR',
    "epoch": 100,
    "img_size": 384,
    "patience": 20
}


def data_transforms_img(img_size):
    data_transforms = {
        "train": A.Compose([
            # 이미지의 maxsize를 max_size로 rescale합니다. aspect ratio는 유지.
            A.LongestMaxSize(max_size=int(img_size * 1.0)),
            # min_size보다 작으면 pad
            A.PadIfNeeded(min_height=int(img_size * 1.0), min_width=int(img_size * 1.0),
                          border_mode=cv2.BORDER_CONSTANT),
            # A.ToGray(p=1),
            # A.Resize(img_size, img_size),
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
            # A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            transforms.ToTensorV2()])
    }
    return data_transforms


def padding(img, set_size):
    try:
        h, w, c = img.shape
    except:
        print('파일을 확인후 다시 시작하세요.')
        raise

    if h < w:
        new_width = set_size
        new_height = int(new_width * (h / w))
    else:
        new_height = set_size
        new_width = int(new_height * (w / h))

    if max(h, w) < set_size:
        img = cv2.resize(img, (new_width, new_height), cv2.INTER_CUBIC)
    else:
        img = cv2.resize(img, (new_width, new_height), cv2.INTER_AREA)

    try:
        h, w, c = img.shape
    except:
        print('파일을 확인후 다시 시작하세요.')
        raise

    delta_w = set_size - w
    delta_h = set_size - h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return new_img


def test_label_encoding(df, encoder):
    df['new_labels'] = 'None'
    for i in range(len(df)):
        if df['labels'][i] in encoder.classes_:
            df['new_labels'][i] = int((np.where(encoder.classes_ == df['labels'][i])[0]))
            df[df['new_labels'] != 'None']
    return df[df['new_labels'] != 'None']


def get_first_img(path):
    for img_path in os.listdir(path):
        return path + img_path


def fetch_scheduler(optimizer):
    if config['sch'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500,
                                                   eta_min=1e-6)
    elif config['sch'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.9, verbose=True)
    # elif sch == 'CosineAnnealingWarmRestarts':
    #     scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'],
    #                                                          eta_min=CONFIG['min_lr'])
    elif config['sch'] == None:
        return None

    return scheduler


if __name__ == "__main__":
    result = []
    train_path = "s3://sagemaker-studio-475719114507-5isnds6vsgd/data/EMB_clean"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #     for AWS .
    bucket = train_path.split('/')[2]
    subfolder = '/'.join(train_path.split('/')[3:])

    conn = boto3.client('s3')
    fs = s3fs.S3FileSystem()

    train_df = aws_make_csv_file('/train/', conn, bucket, subfolder)

    if os.path.isfile('label.json'):
        with open('label.json', 'r') as file:
            label_name = json.load(file)
        encoder = LabelEncoder()
        if len(list(label_name.keys())) == len(train_df['labels']):
            # *Caution*  if json A(before) and json B(now) length are the same. please check both have same file list
            encoder.fit(list(label_name.keys()))
            train_df['new_labels'] = encoder.fit_transform(train_df['labels'])
        else:
            encoder = LabelEncoder()
            train_df['new_labels'] = encoder.fit_transform(train_df['labels'])
            target_encodings = {t: i for i, t in enumerate(encoder.classes_)}
            with open('label.json', 'w') as f:
                json.dump(target_encodings, f)
    else:
        encoder = LabelEncoder()
        train_df['new_labels'] = encoder.fit_transform(train_df['labels'])
        target_encodings = {t: i for i, t in enumerate(encoder.classes_)}
        with open('label.json', 'w') as f:
            json.dump(target_encodings, f)

    encoder = LabelEncoder()
    train_df['new_labels'] = encoder.fit_transform(train_df['labels'])

    valid_df = test_label_encoding(aws_make_csv_file('/test/', conn, bucket, subfolder), encoder)

    target_size = len(encoder.classes_)

    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path=".")

    with open(os.path.join(model_path, 'model.pth'), 'rb') as f:
        model = torch.load(f)
    # model = EMB_model(model_name=config['model_name'], target_size=target_size)
    model.to(device)

    model.eval()

    data_transforms = data_transforms_img(config['img_size'])

    Test = aws_EMB_Dataset(valid_df, fs, transforms=data_transforms['valid'])
    print(valid_df)
    Test_loader = DataLoader(Test, batch_size=2, num_workers=0, shuffle=True, pin_memory=True, drop_last=True)
    # model test
    total = 0
    correct = 0
    name = []
    co = []
    predict = []
    with torch.no_grad():
        model.eval()
        bar = tqdm(enumerate(Test_loader), total=len(Test_loader))
        for step, data in bar:
            img = data['image'].to(device, dtype=torch.float)
            labels = data['new_labels'].to(device, dtype=torch.long)

            # classification
            outputs = model(img)
            # print(outputs)
            # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
            conf, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            if predicted != labels.cpu().detach().numpy():
                name.append(data['path'][0].split('/')[-1])
                collect = encoder.classes_[labels.cpu()]
                model_pred = encoder.classes_[predicted.cpu()]

                co.append(collect)
                predict.append(model_pred)

            correct += (predicted == labels).sum().item()

    logger.info(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    report_dict = {
        "acc_metrics": {
            "acc": {"value": 100 * correct // total, "correct":correct, "total": total},
        },
    }

    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))
