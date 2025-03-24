#!/bin/bash

# 激活minimind环境
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate minimind

echo "minimind环境已激活"
echo "Python版本: $(python --version)"
echo "CUDA是否可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "可用GPU数量: $(python -c 'import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)')"

# 切换到项目目录
cd ~/minimind

# 保持终端在此环境中
exec bash 