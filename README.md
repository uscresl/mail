# Learning Robot Manipulation from Cross-Morphology Demonstration

[[Project website](https://uscresl.github.io/mail/)]
[[arXiv PDF](https://arxiv.org/abs/2304.03833)]

This project is a PyTorch implementation of Learning Robot Manipulation from Cross-Morphology Demonstration, published in CoRL, 2023.

### Setup
[Code](https://github.com/Xingyu-Lin/VCD) is based on the official implementation of Learning Visible Connectivity Dynamics for Cloth Smoothing.

This repository is a subset of [SoftAgent](https://github.com/Xingyu-Lin/softagent) cleaned up for VCD. Environment setup for VCD is similar to that of softagent.
1. Install [SoftGym](https://github.com/Xingyu-Lin/softgym). Then, copy softgym as a submodule in this directory by running `cp -r [path to softgym] ./`. Use the updated softgym on the vcd branch by `cd softgym && git checkout vcd`
2. You should have a conda environment named `softgym`. Install additional packages required by VCD, by `conda env update --file environment.yml` 
3. Generate initial environment configurations and cache them, by running `python VCD/generate_cached_initial_state.py`.
4. Run `./compile_1.0.sh && . ./prepare_1.0.sh` to compile PyFleX and prepare other paths.

### GNS Experiments
1. Create `data` folder for experiments.
```
mkdir data
```

2. Download [checkpoints](https://drive.google.com/drive/folders/1Mnmg2G2OP3Q1ZhzJIaBDlvBvDK4kGtw9?usp=sharing) and place them inside the `data` folder.

3. Download [cached_initial_states](https://drive.google.com/drive/folders/1OXCmi-CzjocA4Rfh0RRgiYz_xCuXlKR2?usp=sharing) and place it inside `gns/softgym/softgym/` folder.
#### Dry Cloth
```
############ Training ############
sh ./scripts/train/gns/drycloth_gen_dataset.sh     # generate random actions dataset
sh ./scripts/train/gns/drycloth_training.sh        # train GNS

############ Evaluation ############
sh ./scripts/eval/gns/drycloth_eval.sh
```

#### Cloth Fold
```
############ Training ############
sh ./scripts/train/gns/clothfold_gen_dataset.sh      # generate random actions dataset
sh ./scripts/train/gns/clothfold_training.sh         # train GNS

############ Evaluation ############
sh ./scripts/eval/gns/clothfold_eval.sh
```

## Troubleshooting

ImportError: libpcl_keypoints.so.1.7: cannot open shared object file: No such file or directory #317.
*  Refer to https://github.com/strawlab/python-pcl/issues/317 for more details.

```
# Install PyTorch with CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# To install PyG, ensure nvcc/CUDA toolkit is installed. You can check if nvcc is installed by executing nvcc --version.

# Installing PyG
CUDA=cu113
TORCH=1.12.0
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric


# How to install python-pcl properly? Just doing pip install python-pcl will get this error: ImportError: libpcl_keypoints.so.1.7: cannot open shared object file: No such file or directory #317. Refer to https://github.com/strawlab/python-pcl/issues/317 for more details

# First, add the following lines to /etc/apt/sources.list
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse

# Run the following in sequence:
sudo apt update 
pip install python-pcl
sudo apt-get install libpcl-keypoints1.7
sudo apt-get install libpcl-outofcore1.7
sudo apt-get install libpcl-people1.7
sudo apt-get install libpcl-recognition1.7
sudo apt-get install libpcl-registration1.7
sudo apt-get install libpcl-segmentation1.7
sudo apt-get install libopenmpi1.10
sudo apt-get install libvtk6.2
sudo apt-get install libpcl-surface1.7
sudo apt-get install libpcl-tracking1.7
sudo apt-get install libflann1.8
sudo apt-get install libpcl-visualization1.7
```

## Cite
If you find this codebase useful in your research, please consider citing:
```
@inproceedings{salhotra2023mail,
      title={Learning Robot Manipulation from Cross-Morphology Demonstration},
      author={Gautam Salhotra and I-Chun Arthur Liu and Gaurav S. Sukhatme},
      booktitle={Conference on Robot Learning},
      year={2023}
}
```