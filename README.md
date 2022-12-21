# scSHARP

Repository for "Consensus Label Propagation with Graph Convolutional Networks for Single-Cell RNA Sequencing Cell Type Annotation" extended abstract submission for Learning on Graphs Conference and full paper available at https://doi.org/10.1101/2022.11.23.517739.

## R tools Installation
You will need one R dependency to run scSHARP. Repository and install instructions can be found [here](https://github.com/W-Holtz/R4scSHARP)

## scSHARP Installation

Install instructions are based on a 
### Create a new conda environment
```
conda create -n <env name> python=3.9
conda activate <env name>
```
### Install pip to conda environment
```
conda install pip
```
### Install torch

Linux with GPU:
```
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
``` 
Mac OS:
```
conda install pytorch -c pytorch
```

### Install torch geometric
```
conda install pyg -c pyg
```

### Install scSHARP
You will need to use the version of pip installed to your new conda environemt. In order to find the path to your conda environment, you can use:
```
conda config --show envs_dirs
```
Run the pip install of scSHARP directly from the binary in yout conda directory
```
./anaconda/envs/<env_name>/bin/pip install scSHARP
```

Installation of torch and torch geometric required prior to pip install.

See demo.ipynb for example work flow.
Please input raw counts.
