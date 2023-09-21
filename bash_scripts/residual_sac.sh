#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=1_5
#SBATCH --output=/hpcwork/thes1499/10_8/robosuite/stable-baselines3/task_out/task_out.%J.out
#SBATCH --error=/hpcwork/thes1499/10_8/robosuite/stable-baselines3/error_out/error_task_out.%J.out
#SBATCH --nodes=1 # request one nodes
#SBATCH --cpus-per-task=8  # ask for 2 cpus per task
#SBATCH --mem=64G
#SBATCH --account=rwth1272
#SBATCH --time=12:00:00
# request one gpu per node 


export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"


# load modules
# module switch intel gcc      
conda activate robosuite

# PYTHONPATH='/hpcwork/ru745256/master_thesis/robosuite-benchmark/rlkit':${PYTHONPATH}
# PYTHONPATH='/hpcwork/ru745256/master_thesis/robosuite-benchmark':${PYTHONPATH}
PYTHONPATH='/hpcwork/thes1499/10_8/robosuite':${PYTHONPATH}
PYTHONPATH='/hpcwork/thes1499/10_8/robosuite/stable-baselines3':${PYTHONPATH}                       
export PYTHONPATH
export MUJOCO_GL='disabled'
#export PYOPENGL_PLATFORM=osmesa
#export DISPLAY=guilinuxbox:0.0


python /hpcwork/thes1499/10_8/robosuite/stable-baselines3/Residual_RL/examples/residual.py task_config.reward_mode=2 seed=9