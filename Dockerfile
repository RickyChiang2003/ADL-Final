FROM ubuntu:20.04

RUN mkdir -p /root/final
RUN apt-get update && apt-get install -y vim libpq-dev wget
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    sh Miniconda3-latest-Linux-x86_64.sh -b -p /root/final/miniconda3 && \
    rm -r Miniconda3-latest-Linux-x86_64.sh
ENV PATH /root/final/miniconda3/bin:$PATH
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda install python=3.12


WORKDIR /root/final
COPY requirements.txt .

RUN /bin/bash -c "source /root/final/miniconda3/bin/activate && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -n adl-final python==3.12 && \
    conda activate adl-final && \
    pip install --upgrade pip && \
    pip install -r requirements.txt"

RUN apt install openssh-server -y
EXPOSE 22
RUN mkdir -p /root/.ssh
COPY ./id_ed25519.pub /root/.ssh/authorized_keys

RUN apt install tmux nano -y
ENTRYPOINT service ssh restart && tail -f /dev/null

EXPOSE 8080

