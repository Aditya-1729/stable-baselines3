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


export CONDA_ROOT=$HOME/miniconda3
. $CONDA_ROOT/etc/profile.d/conda.sh
export PATH="$CONDA_ROOT/bin:$PATH"


# load modules   
conda activate robosuite


PYTHONPATH='/work/thes1499/DR_19_10/robosuite':${PYTHONPATH}
PYTHONPATH='/work/thes1499/DR_19_10/robosuite/stable-baselines3':${PYTHONPATH}                       
export PYTHONPATH
export MUJOCO_GL='disabled'


python /work/thes1499/DR_19_10/robosuite/stable-baselines3/sac_hydra.py algorithm=sac_residual experiment=sac_residual seed=37
# python /work/thes1499/DR_19_10/robosuite/stable-baselines3/sac_hydra.py algorithm=sac_curriculum algorithm.curriculum.steps=10 algorithm.curriculum.complete_handover=0.2 seed=54