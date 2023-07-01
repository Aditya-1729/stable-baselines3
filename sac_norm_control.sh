#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=sac_pos_limit
#SBATCH --output=/hpcwork/ru745256/master_thesis/30_6/robosuite/stable-baselines3/task_out.%J.out
#SBATCH --error=/hpcwork/ru745256/master_thesis/30_6/robosuite/stable-baselines3/error_task_out.%J.out
#SBATCH --nodes=1 # request one nodes
#SBATCH --cpus-per-task=16  # ask for 2 cpus per task
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# request one gpu per node 
#SBATCH --gres=gpu:volta:1


export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"


# load modules
# module switch intel gcc      
conda activate robosuite

# PYTHONPATH='/hpcwork/ru745256/master_thesis/robosuite-benchmark/rlkit':${PYTHONPATH}
# PYTHONPATH='/hpcwork/ru745256/master_thesis/robosuite-benchmark':${PYTHONPATH}
PYTHONPATH='/hpcwork/ru745256/master_thesis/30_6/robosuite':${PYTHONPATH}
PYTHONPATH='/hpcwork/ru745256/master_thesis/30_6/robosuite/stable-baselines3':${PYTHONPATH}                       
export PYTHONPATH
export MUJOCO_GL='disabled'
#export PYOPENGL_PLATFORM=osmesa
#export DISPLAY=guilinuxbox:0.0


python /hpcwork/ru745256/master_thesis/30_6/robosuite/stable-baselines3/sac_robosuite.py 