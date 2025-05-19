FROM nvcr.io/nvidia/pytorch:25.04-py3

RUN pip install --upgrade pip
RUN pip install torch-pruning torchinfo matplotlib onnx

ENV PYTHONPATH=/ava-x/genesis/components/hydra/src/py
