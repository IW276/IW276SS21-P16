FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3

COPY . /app

WORKDIR /app

RUN pip3 install -U pip wheel setuptools && pip3 install -r requirements.txt
