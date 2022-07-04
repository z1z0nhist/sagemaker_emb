import copy
import time

import pandas as pd
import argparse
import os
import json
import numpy as np
import logging
import heapq
from PIL import Image

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import glob
import cv2
import boto3
import s3fs
import json

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import albumentations as A
import gc

from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from albumentations.pytorch import transforms
from utils import EMB_Dataset, data_transforms_img, make_csv_file, test_label_encoding, training_epoch, valid_epoch

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
config = {
    "sch": 'CosineAnnealingLR',
}


class EMB_model(nn.Module):
    def __init__(self, model_name, target_size, pretrained=False):
        bucket_name = 'sagemaker-project-p-pggiw8qb44oo'

        super(EMB_model, self).__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=target_size)

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

        if not pretrained:
            s3 = boto3.resource('s3')
            conn = boto3.client('s3')
            fs = s3fs.S3FileSystem()
            my_bucket = s3.Bucket(bucket_name)
            heap = []
            for label in my_bucket.objects.filter(Prefix='emb-models'):
                if label.key.endswith('.pth'):
                    heapq.heappush(heap, f's3://' + bucket_name + '/' + label.key)
            logger.info('recent model loss is : ' + heap[0])
            with fs.open(heap[0]) as f:
                self.model = torch.load(f, map_location=device)

    def forward(self, x):
        x = self.model(x)
        return x


def fetch_scheduler(optimizer):
    if config['sch'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500, eta_min=1e-6)
    elif config['sch'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.9, verbose=True)
    elif config['sch'] == None:
        return None


def run_training(args, model, optimizer, scheduler, device, num_epochs, Train_loader, test_loader):
    if torch.cuda.is_available():
        logger.info("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_loss = np.inf
    for epoch in range(1, num_epochs + 1):
        gc.collect()
        train_epoch_loss = training_epoch(model, optimizer, scheduler,
                                          dataloader=Train_loader,
                                          device=device, epoch=epoch)

        val_epoch_loss = valid_epoch(model, test_loader, device=device,
                                     epoch=epoch)

        if val_epoch_loss <= best_epoch_loss:
            logger.info(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            save_model(model, args, loss=best_epoch_loss)

        logger.info('\n')

        end = time.time()
        time_elapsed = end - start
        logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        logger.info("Best Loss: {:.4f}".format(best_epoch_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)

    return model, best_epoch_loss


def train(args):
    data_transforms = data_transforms_img(args.img_size)
    # # 파일경로 추적
    # for (path, dir, files) in os.walk('/opt/ml/'):
    #    for filename in files:
    #        if not str(filename).endswith(".jpg", ".JPG"):
    #            logger.info("%s/%s" % (path, filename))
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    # data loader
    logger.info("Loading EMB dataset")

    train_path = os.path.join('/opt/ml/input/data/', args.data_dir)
    valid_path = os.path.join('/opt/ml/input/data/', args.test_dir)

    logger.info(train_path + '/')
    logger.info(args.test_dir + '/')

    train_df = make_csv_file(train_path + '/')
    if os.path.isfile('label.json'):
        with open('label.json', 'r') as file:
            label_name = json.load(file)
        encoder = LabelEncoder()
        if len(list(label_name.keys())) == len(os.listdir(train_path)):
            logger.info('load label.json...')
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
        logger.info('cant load label.json...')
        encoder = LabelEncoder()
        train_df['new_labels'] = encoder.fit_transform(train_df['labels'])
        target_encodings = {t: i for i, t in enumerate(encoder.classes_)}
        with open('label.json', 'w') as f:
            json.dump(target_encodings, f)

    logger.info(train_df)

    # train_df['new_labels'] = encoder.fit_transform(train_df['labels'])
    valid_df = test_label_encoding(make_csv_file(valid_path + '/'), encoder)

    logger.info(valid_df)
    Train = EMB_Dataset(train_df, transforms=data_transforms['train'])
    Test = EMB_Dataset(valid_df, transforms=data_transforms['valid'])
    Train_loader = DataLoader(Train, args.batch_size, shuffle=True)
    test_loader = DataLoader(Test, args.test_batch_size, shuffle=True)
    target_size = len(encoder.classes_)
    # model

    model = EMB_model(model_name=args.model_name, target_size=target_size).to(device)

    logger.info("Model loaded")

    optimizer = optim.Adam(model.parameters(), lr=0.0001,
                           weight_decay=1e-6)
    scheduler = fetch_scheduler(optimizer)

    model, best_epoch_loss = run_training(args, model, optimizer, scheduler,
                                                   device=device,
                                                   num_epochs=args.epochs,
                                                   Train_loader=Train_loader,
                                                   test_loader=test_loader)

    save_model(model, args, loss=best_epoch_loss)


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    test_loss = 0
    correct = 0
    total = 0
    bar = tqdm(enumerate(test_loader), total=len(test_loader))
    with torch.no_grad():
        for step, data in bar:
            img, labels = data['image'].to(device, dtype=torch.float), data['new_labels'].to(device, dtype=torch.long)
            output = model(img)
            loss = criterion(output, labels)
            batch_size = img.size(0)
            test_loss += (loss.item() * batch_size)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += labels.size(0)
    test_loss /= total
    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))
    return {'loss': test_loss, 'acc': correct / total}

def save_model(model, args, loss):
    s3 = boto3.client('s3')
    bucket_name = 'sagemaker-project-p-pggiw8qb44oo'

    path = os.path.join(args.model_dir, 'model.pth')
    logger.info("Saving the model." + path)
    torch.save(model, path)

    s3.upload_file(path, bucket_name, f'emb-models/{args.model_name}/{loss}-model.pth')
    logger.info("Saving the model to S3." + bucket_name + '/emb-models' + f'/{args.model_name}/{loss}-model.pth')


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = EMB_model(model_name='convnext_base_384_in22ft1k', target_size=130).to(device)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 10)'),
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    parser.add_argument('--model_name', type=str, default='convnext_base_384_in22ft1k',
                        help='timm base model name')
    parser.add_argument('--img_size', type=int, default=448,
                        help='model train img size')
    parser.add_argument('--sch', type=str, default='CosineAnnealingLR',
                        help='lr sch arg')

    # Container environment
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    # parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    # parser.add_argument('--test_dir', type=str, default=os.environ.get('SM_CHANNEL_TESTING'))
    parser.add_argument("--data_dir", type=str, default=os.environ.get("SM_CHANNEL_TRAINING"))
    parser.add_argument("--test_dir", type=str, default=os.environ.get("SM_CHANNEL_TESTING"))
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    train(parser.parse_args())

