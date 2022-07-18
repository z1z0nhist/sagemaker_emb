ARG REGION=us-west-1
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

COPY code/requirements.txt /build/

ENV PATH="/opt/ml/code:${PATH}"

RUN pip install --upgrade pip
RUN pip install -r /build/requirements.txt

# Install sagemaker-training toolkit to enable SageMaker Python SDK
RUN pip3 install sagemaker-training

# Copies the training code inside the container
#COPY train.py /opt/ml/code/train.py
COPY /code /opt/ml/code
#COPY /code /opt/program/serve

#RUN mkdir -p /opt/ml/model/code
#COPY requirements.txt /opt/ml/model/code/
#COPY inference.py /opt/ml/model/code/

# Defines train.py as script entrypoint
#ENTRYPOINT ["python", "/opt/ml/code/train.py"]
# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train.py