# all tf dockers here https://hub.docker.com/r/tensorflow/tensorflow/tags/?page=1&ordering=last_updated
# nothing for 2.3 !!!
ARG REGION=us-west-1
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

COPY requirements.txt /build/

RUN pip install --upgrade pip
RUN pip install -r /build/requirements.txt

# Install sagemaker-training toolkit to enable SageMaker Python SDK
RUN pip3 install sagemaker-training

# Copies the training code inside the container
#COPY train.py /opt/ml/code/train.py
COPY . /opt/ml/code/

# Defines train.py as script entrypoint
ENTRYPOINT ["python", "/opt/ml/code//train.py"]
