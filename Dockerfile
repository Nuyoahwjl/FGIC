# FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel
FROM docker.m.daocloud.io/pytorch/pytorch:2.6.0-cuda11.8-cudnn9-devel

WORKDIR /workspace

ENV TZ=Asia/Shanghai
ENV LANG=C.UTF-8

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list && \
    sed -i 's/security.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list

RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn && \
    pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

COPY source/requirements.txt .
COPY source/main.py .
COPY source/entry.sh .
COPY source/configs ./configs
COPY source/src ./src

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /.cache /.config && \
    chmod -R 777 /.cache /.config

CMD ["bash", "entry.sh"]