FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y git wget \
    libopencv-dev autoconf automake libtool

RUN cd /usr/local/src && \
    git clone -b v2.0 https://github.com/blitzingeagle/darknet.git --recurse-submodules && \
    cd darknet/json-c && \
    sh autogen.sh && ./configure && make && make install && make check

ENV DARKNET_PATH /usr/local/src/darknet
WORKDIR $DARKNET_PATH

RUN cd $DARKNET_PATH && make && \
    cd /usr/local/bin && ln -s $DARKNET_PATH/darknet darknet
