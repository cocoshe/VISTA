#!/bin/bash

# install torch
# pip install torch==2.5.0 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# install FA2 and diffusers
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation 

pip install -r requirements-lint.txt

pip install -r requirements.txt

# install fastvideo
pip install -e .
