#!/usr/local_rwth/bin/zsh

#SBATCH --job-name=Res_37
#SBATCH --output=/work/thes1499/DR_19_10/robosuite/stable-baselines3/task_out/task_out.%J.out
#SBATCH --error=/work/thes1499/DR_19_10/robosuite/stable-baselines3/error_task/error_task_out.%J.out
#SBATCH --nodes=1 # request one nodes
#SBATCH --cpus-per-task=8  # ask for 2 cpus per task
#SBATCH --mem=64G
#SBATCH --account=rwth1458
#SBATCH --time=17:00:00
# SBATCH --gres=gpu:volta:1

# request one gpu per node 


export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"


# load modules
# module switch intel gcc      
conda activate robosuite

# PYTHONPATH='/hpcwork/ru745256/master_thesis/robosuite-benchmark/rlkit':${PYTHONPATH}
# PYTHONPATH='/hpcwork/ru745256/master_thesis/robosuite-benchmark':${PYTHONPATH}
PYTHONPATH='/work/thes1499/DR_19_10/robosuite':${PYTHONPATH}
PYTHONPATH='/work/thes1499/DR_19_10/robosuite/stable-baselines3':${PYTHONPATH}                       
export PYTHONPATH
export MUJOCO_GL='disabled'
#export PYOPENGL_PLATFORM=osmesa
#export DISPLAY=guilinuxbox:0.0


python /work/thes1499/DR_19_10/robosuite/stable-baselines3/Residual_RL/examples/residual.py algorithm=sac_residual task_config=force_control_4 controller.agent_config=residual_2 experiment=sac_residual seed=37