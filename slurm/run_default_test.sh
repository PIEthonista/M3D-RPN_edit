#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J default_test
#SBATCH -p gp4d
#SBATCH -e slurm_txts/default_test.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

# 558607
# python3
# import torch
# torch.__version__              --> '1.10.1+cu102'
# torch.version.cuda             --> '10.2'
# torch.backends.cudnn.version() --> 7605
# module avail cuda
# module show cuda/10.2          --> setenv("CUDA_HOME","/work/HPC_SYS/nvidia/cuda/cuda-10.2")
# # default variable will not work: /work/HPC_SYS/nvidia/cuda/cuda-10.2
export CUDAHOME=/work/HPC_SYS/nvidia/cuda/cuda-10.2
cd lib/nms
make
cd ../..
python scripts/test_rpn_3d.py
