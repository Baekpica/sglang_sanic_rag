ARG CUDA_VERSION=12.5.1

FROM nvcr.io/nvidia/tritonserver:24.04-py3-min

ARG BUILD_TYPE=all
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa -y && apt update \
    && apt install python3.10 python3.10-dev -y \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --set python3 /usr/bin/python3.10 && apt install python3.10-distutils -y \
    && apt install curl git sudo libibverbs-dev -y \
    && apt install -y rdma-core infiniband-diags openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 cmake libopenblas-dev \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py \
    && python3 --version \
    && python3 -m pip --version \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 -ms /bin/bash appuser

RUN python3 -m pip install --no-cache-dir --upgrade \
    pip \
    uv \
    virtualenv 

WORKDIR /home/appuser/app

COPY . .

ARG CUDA_VERSION
ENV VIRTUAL_ENV=/home/appuser/app/venv
RUN virtualenv ${VIRTUAL_ENV}
RUN uv pip install --no-cache-dir -r ./requirements.txt \
    && uv pip install --no-cache-dir pip setuptools wheel html5lib six \
    && uv pip install --no-cache-dir datamodel_code_generator \
    && uv pip install "sglang[all]>=0.4.3.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer-python

ENV DEBIAN_FRONTEND=interactive
ENV PATH="/home/appuser/app/venv/bin:$PATH"

EXPOSE 30000

ENTRYPOINT ["/home/appuser/app/venv/bin/python3", "app.py"]
