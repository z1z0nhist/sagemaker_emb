import os
import json
import numpy as np
import logging

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import gc

import boto3
import s3fs
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from sagemaker_inference import (
    content_types,
    decoder,
    default_inference_handler,
    encoder,
    errors,
    utils,
)

INFERENCE_ACCELERATOR_PRESENT_ENV = "SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.getLevelName("INFO"),
    handlers=[logging.StreamHandler(sys.stdout)],
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EMB_model(nn.Module):
    def __init__(self, model_name, target_size, pretrained=False):
        super(EMB_model, self).__init__()
        self.model = timm.create_model(model_name=model_name, pretrained=pretrained, num_classes=target_size)

    def forward(self, x):
        x = self.model(x)
        return x


# def input_fn(input_data, content_type):
#     """A default input_fn that can handle JSON, CSV and NPZ formats.

#     Args:
#         input_data: the request payload serialized in the content_type format
#         content_type: the request content_type

#     Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor,
#         depending if cuda is available.
#     """
#     logger.info("input_fn. \n")
#     logger.info(type(input_data)+type(content_type))
#     logger.info(input_data)
#     logger.info(content_type)
#     np_array = decoder.decode(input_data, content_type)
#     tensor = torch.FloatTensor(
#         np_array) if content_type in content_types.UTF8_TYPES else torch.from_numpy(np_array)

#     logger.info(tensor)
#     return tensor.to(device)

def input_fn(input_data, content_type):
    """A default input_fn that can handle JSON, CSV and NPZ formats.

    Args:
        input_data: the request payload serialized in the content_type format
        content_type: the request content_type

    Returns: input_data deserialized into torch.FloatTensor or torch.cuda.FloatTensor depending if cuda is available.
    """
    logger.info("input_fn. \n")
    #         logger.info(input_data)
    #         logger.info(content_type)
    return decoder.decode(input_data, content_type)


def model_fn(model_dir):
    logger.info("Load Model. \n" + os.path.join(model_dir, 'model.pt'))
    model = None
    try:
        with open(os.path.join(model_dir, 'model.pt'), 'rb') as f:
            model = torch.jit.load(f, map_location=device)
    except:
        logger.error("model file not found. " + os.path.join(model_dir, 'model.pt'))
    model.eval()
    return model


def predict_fn(data, model):
    logger.info('Predict_fn...')
    logger.info(type(data))
    data = torch.Tensor(data)
    input_data = data.to(device, dtype=torch.float)

    logger.info(type(input_data))
    logger.info(input_data.shape)
    #     logger.info(model)
    with torch.no_grad():
        output = model(input_data)
    return output


def output_fn(prediction, accept):
    """A default output_fn for PyTorch. Serializes predictions from predict_fn to JSON, CSV or NPY format.

    Args:
        prediction: a prediction result from predict_fn
        accept: type which the output data needs to be serialized

    Returns: output data serialized
    """
    logger.info("Output_fn...")

    bucket_name = 'sagemaker-project-p-pggiw8qb44oo'
    s3 = boto3.resource('s3')
    fs = s3fs.S3FileSystem()

    my_bucket = s3.Bucket(bucket_name)
    json_path = ''
    for label in my_bucket.objects.filter(Prefix='emb-models'):
        if label.key.endswith('.json'):
            json_path = f's3://{bucket_name}/{label.key}'
            break
    logger.info(json_path)
    with fs.open(json_path) as f:
        label_name = json.load(f)
        label_encoder = LabelEncoder()
        label_encoder.fit(list(label_name.keys()))

    logger.info(type(prediction))
    # 가장 높은 값(energy)를 갖는 분류(class)를 정답으로 선택하겠습니다
    conf, predicted = torch.max(prediction.data, 1)
    logger.info(conf)
    logger.info(predicted)

    model_pred = label_encoder.classes_[predicted.cpu()]
    logger.info(model_pred)
    #     if type(prediction) == torch.Tensor:
    #         prediction = prediction.detach().cpu().numpy().tolist()

    for content_type in utils.parse_accept(accept):
        if content_type in encoder.SUPPORTED_CONTENT_TYPES:
            encoded_prediction = encoder.encode(
                {"result": model_pred,
                 "conf": conf.cpu().item() * 0.1}, content_type)
            if content_type == content_types.CSV:
                encoded_prediction = encoded_prediction.encode("utf-8")
            return encoded_prediction

    return encoded_prediction.encode("utf-8")
