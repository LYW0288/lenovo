#FROM nvcr.io/nvidia/tensorflow:20.03-tf2-py3
#FROM nvcr.io/nvidia/l4t-pytorch:r32.5.0-pth1.7-py3
FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install cron -y && \
    apt-get install openssh-server -y && \
    apt-get install net-tools -y && \
    apt-get install wget -y && \
    apt-get install vim -y && \
    apt-get install curl -y && \
#    apt-get install python3-pip -y && \
    apt-get install nano -y && \
    apt-get install unixodbc unixodbc-dev -y && \
    apt-get install libsm6 -y && \
    apt-get install libxrender1 -y && \
    apt-get install libxext-dev -y
#    apt-get install python3.7-dev -y 
RUN apt-get update && apt-get upgrade -y
#RUN rm /usr/bin/python && \ 
#    ln -s /usr/bin/python3.7 /usr/bin/python && \ 
#    wget https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \ 
#    python get-pip.py.1
# setting ssh
RUN echo "PermitRootLogin yes" >> /etc/ssh/sshd_config

#set image time zone
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata 

RUN TZ=Asia/Taipei && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata 


ENV PATH="/opt/ml/code:${PATH}"
ARG PROJECT_NAME="${PROJECT_NAME}"


## install cmake 3.17 && opencv
RUN apt-get remove cmake -y
RUN apt-get update \
 && apt-get install -y --no-install-recommends --allow-unauthenticated \
#    libopencv-dev \
#    libopencv-contrib-dev \
    python3-lxml \
    libarchive-tools \
    unzip

ADD requirements.txt /opt
RUN pip3 install -r /opt/requirements.txt
    
RUN curl -sSL https://github.com/Kitware/CMake/releases/download/v3.17.2/cmake-3.17.2-Linux-x86_64.tar.gz | tar -xzC /opt

RUN mv /opt/cmake-3.17.2-Linux-x86_64 /opt/cmake
RUN ln -s /opt/cmake/bin/cmake /usr/bin/cmake

## fix /usr/local/cuda-10.0/compat/libcuda.so
RUN bash -c 'echo "/usr/local/cuda-10.0/compat" > /etc/ld.so.conf.d/cuda.conf'
RUN ldconfig -v


#RUN cd /opt && git clone https://github.com/WongKinYiu/yolov7

RUN apt install python-is-python3 -y

WORKDIR /opt/program

COPY container /opt/program
WORKDIR /opt/program
# entry point for sagemaker training job 
ENTRYPOINT ["python3", "server_validation.py", "--model_path", "./best.pt", "--val_dir", "./data/", "--log_file", "./log.log", "--out_path", "./output/"]

# Set up the program in the image
# COPY decision_trees /opt/program
# WORKDIR /opt/program

