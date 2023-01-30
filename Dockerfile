From nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive
#RUN apt update -y 
#RUN apt-get install -y wget
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb
COPY cuda-keyring_1.0-1_all.deb cuda-keyring_1.0-1_all.deb

RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN sed -i '/developer\.download\.nvidia\.com\/compute\/cuda\/repos/d' /etc/apt/sources.list
RUN apt-get update -y 

# language
RUN apt-get update -y
#RUN apt-get install -y language-pack-ja-base language-pack-ja
ENV LC_ALL ja_JP.UTF-8
# timezone
RUN    apt-get install -y tzdata \
    && apt-get clean
ENV TZ Asia/Tokyo

# for development
RUN apt install -y libsm6 libxrender1 libxext6
RUN apt install -y x11-xserver-utils
CMD xhost si:localuser:root && /bin/bash
ENV QT_X11_NO_MITSHM 1

WORKDIR /opt/app/codes/
COPY codes/requirements.txt /opt/app/codes/requirements.txt
# COPY codes/install.sh /opt/app/codes/install.sh
# RUN sh ./install.sh

RUN apt-get install -y build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev zlib1g-dev openssl libffi-dev python3-dev python3-setuptools wget curl

WORKDIR downloads
RUN wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tar.xz
RUN tar xvf Python-3.7.0.tar.xz
RUN  cd Python-3.7.0 && ./configure && make altinstall

RUN ln -s /usr/local/bin/python3.7 /usr/local/bin/python && ln -s /usr/local/bin/python3.7 /usr/local/bin/python3

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

RUN apt install -y libgeos-dev
# RUN cd lib && make clean && make && cd deform_conv && python setup.py develop
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv BA9EF27F
RUN echo 'deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu xenial main' | tee /etc/apt/sources.list.d/ubuntu-toolchain-r-ubuntu-test-xenial.list
# RUN apt update -y && apt install gcc-10 g++-10 -y
RUN apt-get install -y liblzma-dev

#COPY codes/cocoapi cocoapi
WORKDIR /opt/app/
RUN apt install -y git
COPY codes/requirements.txt requirements.txt
RUN pip --version
RUN pip install -r requirements.txt 
RUN rm requirements.txt
#RUN git clone https://github.com/cocodataset/cocoapi.git
#RUN cd cocoapi/PythonAPI &&  python setup.py install --user
#RUN pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
RUN git clone https://github.com/philferriere/cocoapi.git
RUN cd cocoapi/PythonAPI &&  python setup.py install --user
