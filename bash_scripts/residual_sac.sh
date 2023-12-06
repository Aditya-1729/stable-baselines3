#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=ep-env-test
#SBATCH --output=/home/ep652816/grinding_robot/task_out/task_out.%J.out
#SBATCH --error=/home/ep652816/grinding_robot/error_task/error_task_out.%J.out
#SBATCH --nodes=1 # request one nodes
#SBATCH --cpus-per-task=8  # ask for 2 cpus per task
#SBATCH --mem=64G
#SBATCH --time=00:05:00
#SBATCH --gres=gpu:volta:1

# request one gpu per node 


module load GCCcore/.9.3.0
module load cuDNN/8.6.0.163-CUDA-11.8.0

export CONDA_ROOT=$HOME/miniconda3
source $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"

conda activate robosuite

# PYTHONPATH='/hpcwork/ru745256/master_thesis/robosuite-benchmark/rlkit':${PYTHONPATH}
# PYTHONPATH='/hpcwork/ru745256/master_thesis/robosuite-benchmark':${PYTHONPATH}
PYTHONPATH='/home/ep652816/grinding_robot/robosuite/stable-baselines3':${PYTHONPATH}
PYTHONPATH='/home/ep652816/grinding_robot/robosuite':${PYTHONPATH}                       
export PYTHONPATH
export MUJOCO_GL='disabled'
#export PYOPENGL_PLATFORM=osmesa
#export DISPLAY=guilinuxbox:0.0


cd /home/ep652816/grinding_robot

python robosuite/stable-baselines3/Residual_RL/examples/residual.py algorithm=sac_residual task_config=force_control_new_reward controller.agent_config=residual_2 experiment=new-reward-debug seed=37
