+# IDPG
IDPG: An Instance-Dependent Prompt Generation Method

This repository contains the code for the paper [IDPG: An Instance-Dependent Prompt Generation Method](https://arxiv.org/abs/2204.04497)

**************************** **Updates** ****************************

* 4/23: Authors released [training code](#training).
* 4/9: Authors [our paper](https://arxiv.org/pdf/2204.04497.pdf)
* 4/7: Paper has been accepted to NAACL 2022 main conference!

## Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install IDPG** and develop locally:

```bash
conda create -n IDPG python=3.6
conda activate IDPG
git clone https://github.com/kguo1/IDPG.git
cd IDPG 
pip install --editable ./
pip install requests
pip install pytorch-metric-learning==0.9.90.dev0

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# on A100:
# conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
# pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Pre-trained models
We mainly trained our model starting from below two checkpoints. We used both roberta.large for the fractional prompt insertion and roberta.base for the multi prompt insertion setup. 

Model | Description | # params | Download
---|---|---|---
`roberta.large` | RoBERTa using the BERT-large architecture | 355M | [roberta.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz)

`roberta.base` | RoBERTa using the BERT-base architecture | 
125M | [roberta.base.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.base.tar.gz)


### data
The data bin folders can be downloaded from this link; https://www.dropbox.com/scl/fi/lwgwgypmotrfs9zolpunl/68611projdata.zip?rlkey=uimf2bwxy41klkgs83d5yjrvq&st=tw1aohoy&dl=0. Downlaod the zip file, place it in the IDPG directory, and unzip the file contents inside of the folder.

## Train and evaluate IDPG Fractional prompt insertion
We provide example training scripts for fractional prompt insertion. 



### Example usage
Take The example below as an example. To run it, you need to download checkpoints. Next, modify the `SAVE` and `NEWSAVE` variable, where `SAVE` is the path to your checkpoint folder, `NEWSAVE` is the path where you hope to store the intermediate checkpoints (i.e., fine-tuned checkpoints). 

We explain the main arguments in following:
* `insertionpositions`: The prompt insertion mode. modes 0 through four correspond to the prompt insertion positions from the original IDPG paper. mode 5 corresponds to fractional prompt insertion. 
* `suffixlen`: the generated prompt length in each Transformer layer.
* `LRs`: the tuned learning rate.
* `seeds`: the multiple random seeds.
* `pdim`: the hidden layer size in phm bottleneck  
* `custom_insert_position_fraction`: The prompt insertion fraction F
* `TASKs`: The dataset used for training and evaluation

The result will be stored at the path `OUT_FILE`.

Below is an example of showing you how to train and evaluate Fractional Prompt Insertion:
```
#!/bin/bash
#SBATCH --mail-user=aoejile@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive

HOME_real=/nobackup/users/$(whoami)
HOME=/home/aoejile
PYTHON_VIRTUAL_ENVIRONMENT=IDPG
CONDA_ROOT=$HOME_real/anaconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
echo "Current Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
#export HOROVOD_GPU_ALLREDUCE=MPI
#export HOROVOD_GPU_ALLGATHER=MPI
#export HOROVOD_GPU_BROADCAST=MPI
#export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

## Horovod execution

##CVSIM TEST RUN SUBMISSION SCRIPT
#horovodrun -np $SLURM_NTASKS -H `cat $NODELIST` python /home/aoejile/G-Transformer-Internal/CVSIM-2-Box-GTransformer-MLHC/Models/tune_categ_contin.py -ds cvsim -m transformer -d cuda -c 0 -f cvsim_ijcai_config_cf1
#python /home/aoejile/G-Transformer-Internal/CVSIM-2-Box-GTransformer-MLHC/Models/tune_categ_contin.py -ds cvsim -m transformer -d cuda -c 0 -f cvsim_ijcai_config_cf1

ARCH=roberta_large

SAVE=/home/aoejile/IDPG/scratch/test_run/
NEWSAVE=home/aoejile/IDPG/scratch/test_run/
# for me, i set a same path
#SAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/
#NEWSAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/


ROBERTA_PATH=$SAVE'roberta_large_checkpoint.pt'
#ROBERTA_PATH=$SAVE'nli_large_checkpoint.pt'
#suffixlens="5"
#insertpositions=$1
#simply=$2
#LR="5e-4"

insertpositions="5"
echo $insertpositions "insert positions"
suffixlen="5"
LRs="5e-3 1e-3 5e-4 1e-4"
LRs="5e-4"
seeds="1 2 3 4 5"
seeds="1"

pdim="16"
mode="1"
custom_insert_position_fraction=".9"
#SBATCH -J myjob_${custom_insert_position_fraction}_4GPUs
#SBATCH -o myjob_${custom_insert_position_fraction}_4GPUs_%j.out
#SBATCH -e myjob_${custom_insert_position_fraction}_4GPUs_%j.err
#mkdir -p "main_results"
OUT_FILE='main_results/IDPG-PHM-p-layerb.txt'$pdim'-'$suffixlen

for LR in $LRs; do
    for insertposition in $insertpositions; do
        SUFFIX='-multi-phm-p-layerb-'$mode'-'$pdim'-'$suffixlen'-'$insertposition'-'$custom_insert_position_fraction'-f'$LR'_5-'
        TASKs='mpqa'
        TASKs='rte'
        for TASK in $TASKs; do
            echo $TASK  "task"
            
            for seed in $seeds; do
                echo $seed "random seed"
                 node=0
                SAVE_FILE=$NEWSAVE$TASK$SUFFIX$seed
                echo $custom_insert_position_fraction "custom insert position fraction"
                bash 'scripts/multi-suffix-'$TASK'_finetune-phm-p-layerb.sh' $ROBERTA_PATH $SAVE_FILE $seed $pdim $node $suffixlen $insertposition $LR $mode $custom_insert_position_fraction
            done
            wait
            for seed in $seeds; do
                 SAVE_FILE=$NEWSAVE$TASK$SUFFIX$seed'/'
                 CUDA_VISIBLE_DEVICES=0 python 'scripts/'$TASK'_get_result.py' -i $SAVE_FILE -o $OUT_FILE -n $seed -t $insertposition -t2 $custom_insert_position_fraction -l $LR 
            done
            wait
            #SAVE_FILE=$NEWSAVE$TASK$SUFFIX
            #python 'scripts/bagging_'$TASK'.py' -i $SAVE_FILE -o $OUT_FILE
            #echo $TASK 'done'
        done
    done 
done





echo "Run completed at:- "
date
```



## Train and evaluate IDPG multi position prompt insertion
We provide example training scripts for multi position prompt insertion. 



### Example usage
Take The example below as an example. To run it, you need to download checkpoints. Next, modify the `SAVE` and `NEWSAVE` variable, where `SAVE` is the path to your checkpoint folder, `NEWSAVE` is the path where you hope to store the intermediate checkpoints (i.e., fine-tuned checkpoints). 

We explain the main arguments in following:
* `insertionpositions`: The prompt insertion mode. modes 0 through four correspond to the prompt insertion positions from the original IDPG paper. mode 6 corresponds to multi position prompt insertion. 
* `suffixlen`: the generated prompt length between each input token in each Transformer layer.
* `LRs`: the tuned learning rate.
* `seeds`: the multiple random seeds.
* `pdim`: the hidden layer size in phm bottleneck  
* `custom_insert_position_fraction`: The prompt insertion fraction F
* `distributed_world_size`: The number of gpus available for distributed training
* `batch_size`: The batch size for training
* `max_tokens`: The maximum number of tokens in a batch
* `embed_dim`: The model embedding dim
* `ffn_dim`: The dimensions of the feedforward function
* `ffn_layers`: The number of layers in the feedforward function
* `TASKs`: The dataset used for training and evaluation
* `node`: the cuda visible devices for training. 

The result will be stored at the path `OUT_FILE`.

Below is an example of showing you how to train and evaluate Multi prompt insertion:
```
#!/bin/bash
#SBATCH -J myjob_base_4GPUs
#SBATCH -o myjob_base_4GPUs_%j.out
#SBATCH -e myjob_base_4GPUs_%j.err
#SBATCH --mail-user=aoejile@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclusive

HOME_real=/nobackup/users/$(whoami)
HOME=/home/aoejile
PYTHON_VIRTUAL_ENVIRONMENT=IDPG
CONDA_ROOT=$HOME_real/anaconda3

source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
echo "Current Conda environment: $CONDA_DEFAULT_ENV"
echo "Python path: $(which python)"
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
#export HOROVOD_GPU_ALLREDUCE=MPI
#export HOROVOD_GPU_ALLGATHER=MPI
#export HOROVOD_GPU_BROADCAST=MPI
#export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

## Horovod execution

##CVSIM TEST RUN SUBMISSION SCRIPT
#horovodrun -np $SLURM_NTASKS -H `cat $NODELIST` python /home/aoejile/G-Transformer-Internal/CVSIM-2-Box-GTransformer-MLHC/Models/tune_categ_contin.py -ds cvsim -m transformer -d cuda -c 0 -f cvsim_ijcai_config_cf1
#python /home/aoejile/G-Transformer-Internal/CVSIM-2-Box-GTransformer-MLHC/Models/tune_categ_contin.py -ds cvsim -m transformer -d cuda -c 0 -f cvsim_ijcai_config_cf1

ARCH=roberta_base

SAVE=/nobackup/users/aoejile/IDPG/scratch/test_run/
NEWSAVE=/nobackup/users/aoejile/IDPG/scratch/test_run/
# for me, i set a same path
#SAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/
#NEWSAVE=/scratch/vgvinodv_root/vgvinodv1/zhuofeng/checkpoints/


ROBERTA_PATH=$SAVE'roberta_base_checkpoint.pt'
#ROBERTA_PATH=$SAVE'nli_large_checkpoint.pt'
#suffixlens="5"
#insertpositions=$1
#simply=$2
#LR="5e-4"

insertpositions="6"
echo $insertpositions "insert positions"
suffixlen="5"
##LRs="5e-3 1e-3 5e-4 1e-4"
LRs="5e-4"
seeds="1 2"
##seeds="1"

pdim="16"
mode="1"
custom_insert_position_fraction="1"
distributed_world_size="4"
batch_size="16"
max_tokens="3200"
embed_dim="768"
ffn_dim="3072"
ffn_layers="12"
#SBATCH -J myjob_${custom_insert_position_fraction}_4GPUs
#SBATCH -o myjob_${custom_insert_position_fraction}_4GPUs_%j.out
#SBATCH -e myjob_${custom_insert_position_fraction}_4GPUs_%j.err
#mkdir -p "main_results"
OUT_FILE='main_results/IDPG-base-PHM-p-layerb.txt'$pdim'-'$suffixlen

for LR in $LRs; do
    for insertposition in $insertpositions; do
        SUFFIX='-multi-phm-p-layerb-'$mode'-'$pdim'-'$suffixlen'-'$insertposition'-'$custom_insert_position_fraction'-f'$LR'_5-'
        ##TASKs='mpqa'
        TASKs='cr'
        for TASK in $TASKs; do
            echo $TASK  "task"
            
            for seed in $seeds; do
                echo $seed "random seed"
                 
                 node=0,1,2,3
		 ##node=0
                SAVE_FILE=$NEWSAVE$TASK$SUFFIX$seed
                echo $custom_insert_position_fraction "custom insert position fraction"
                echo $ffn_layers "ffn layers 1"
                bash 'scripts/multi-suffix-'$TASK'_finetune-phm-p-layerb.sh' $ROBERTA_PATH $SAVE_FILE $seed $pdim $node $suffixlen $insertposition $LR $mode $custom_insert_position_fraction $distributed_world_size $batch_size $max_tokens $embed_dim $ffn_dim $ffn_layers
            done
            wait
            for seed in $seeds; do
                 SAVE_FILE=$NEWSAVE$TASK$SUFFIX$seed'/'
                 ##CUDA_VISIBLE_DEVICES=0 python 'scripts/'$TASK'_get_result.py' -i $SAVE_FILE -o $OUT_FILE -n $seed -t $insertposition -t2 $custom_insert_position_fraction -l $LR 
                 python 'scripts/'$TASK'_get_result.py' -i $SAVE_FILE -o $OUT_FILE -n $seed -t $insertposition -t2 $custom_insert_position_fraction -l $LR 
            done
            wait
            #SAVE_FILE=$NEWSAVE$TASK$SUFFIX
            #python 'scripts/bagging_'$TASK'.py' -i $SAVE_FILE -o $OUT_FILE
            #echo $TASK 'done'
        done
    done 
done


echo "Run completed at:- "
date
```
## Training Environments
The codebase is based on [fairseq](https://github.com/pytorch/fairseq). We tested our codes in an Nvidia V100 environment. The original authors notice that the model's performance is sensitive to one's server environment and package version. You may find a slight performance difference if you do not have the exact same environment. They highly recommend you run hyper-parameter tuning in your own environment based on our sample scripts. 

## Original IDPG paper
Please cite the original paper if you use IDPG in your work:
```bibtex
@article{wu2022idpg,
  title={IDPG: An Instance-Dependent Prompt Generation Method},
  author={Wu, Zhuofeng and Wang, Sinong and Gu, Jiatao and Hou, Rui and Dong, Yuxiao and Vydiswaran, VG and Ma, Hao},
  journal={arXiv preprint arXiv:2204.04497},
  year={2022}
}
```
