FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y git wget \
    libopencv-dev

ENV DARKNET_PATH /usr/local/src/darknet
COPY . $DARKNET_PATH
WORKDIR $DARKNET_PATH

RUN cd $DARKNET_PATH && make && \
    cd /usr/local/bin && ln -s $DARKNET_PATH/darknet darknet

