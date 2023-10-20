# v1.02
FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04

ENV TZ=Asia/Dubai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

SHELL ["/bin/bash", "-c"]

RUN /bin/echo -e "Acquire::http::Timeout \"300\";\n\
Acquire::ftp::Timeout \"300\";" >> /etc/apt/apt.conf.d/99timeout

RUN apt update && apt dist-upgrade -y
RUN apt install build-essential libbz2-dev libdb-dev \
  libreadline-dev libffi-dev libgdbm-dev liblzma-dev \
  libncursesw5-dev libsqlite3-dev libssl-dev \
  zlib1g-dev uuid-dev tk-dev wget liblapack-dev \
  graphviz fonts-humor-sans git ffmpeg -y

# install python
RUN VER=3.10.10 \
    && wget "https://www.python.org/ftp/python/$VER/Python-$VER.tgz" \
    && tar -xzvf Python-$VER.tgz \
    && cd Python-$VER \
    && ./configure --enable-optimizations --with-lto \
    && make \
    && make install

# install pip
RUN wget "https://bootstrap.pypa.io/get-pip.py" \
    && python3 get-pip.py

RUN pip install -U pip setuptools wheel
RUN pip install -U jupyter
RUN pip install numpy matplotlib seaborn umap-learn optuna pandas scipy sktime \
                mlflow boto3 streamlit scikit-learn opencv-python POT
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install torchviz pytorch-ignite captum
RUN pip install timm einops
RUN pip install wilds
RUN pip install transformers sentencepiece
RUN pip install ultralytics
RUN pip install ffmpeg-python
RUN pip install moviepy

RUN cat /usr/local/lib/python3.10/site-packages/torch/nn/modules/upsampling.py | grep recompute_scale_factor
COPY ./src/upsampling.py /usr/local/lib/python3.10/site-packages/torch/nn/modules/upsampling.py
RUN cat /usr/local/lib/python3.10/site-packages/torch/nn/modules/upsampling.py | grep recompute_scale_factor