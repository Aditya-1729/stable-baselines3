import wandb
import robosuite as suite
from stable_baselines3 import SAC
from robosuite.wrappers import StandaloneWrapper
from robosuite.wrappers import ResidualWrapper
from robosuite.wrappers import CurriculumWrapper
from robosuite.wrappers import SplitWrapper
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from robosuite.residual_rl.src.policies import ResidualSAC
from time_curriculum.src.tbc.tbc import get_tbc_algorithm
from Callbacks.test import PerformanceLog, Training_info
from stable_baselines3.common.logger import configure
import numpy as np
from termcolor import colored

@hydra.main(version_base=None, config_path="../robosuite/main/config/", config_name="main")
def main(cfg: DictConfig):
    """
    This function uses config file to set up the environment and the model. It also sets up the logger and the callbacks.
    The choice of algorithms is limited to SAC, SAC with Time Based Curriculum, SAC with Residual Policy and SAC with Split Policy.
    Raises an assertion error if the algorithm is not supported.
    Args: 
    cfg: DictConfig: configuration file    
    """

    if cfg.use_wandb:
        run = wandb.init(
            config=cfg,
            project=cfg.project,
            name=cfg.experiment,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            monitor_gym=False,  # auto-upload the videos of agents playing the game
            save_code=False,  # optional
        )

    # setup base environment with the agent configuration corresponding to the chosen algorithm
    cfg.controller.agent_config = cfg.algorithm.name
    base_env = suite.make(
        env_name=cfg.env.name,
        **cfg.env.specs,
        task_config=OmegaConf.to_container(cfg.task_config),
        controller_configs=OmegaConf.to_container(cfg.controller)
    )

    # A wrapped environment and model is setup based on the chosen algorithm
    if cfg.algorithm.name == "Curriculum_SAC":
        wrapped_env = CurriculumWrapper(base_env)
        n = cfg.algorithm.curriculum.steps

        # maximum horizon allowed for the base controller (guide policy)
        max_horizon = wrapped_env.env.horizon

        # freq=(%of total timesteps for complete handover)*total timesteps/number of discrete curriculum steps
        cfg.algorithm.curriculum.freq = int(cfg.algorithm.curriculum.complete_handover * cfg.algorithm.learn.total_timesteps / n)
        print('Complete handover in', colored(f'{int(cfg.algorithm.curriculum.freq * n)}', 'green'), ' time steps with freq:', colored(f'{int(cfg.algorithm.curriculum.freq)}', 'yellow'))
        print('Curriculum steps: ', colored(f'{n}', 'green'))
        print('Guide Policy Curriculum:', colored(f'{np.arange(max_horizon, -1, -max_horizon // n)}', 'green'))

        model = get_tbc_algorithm(SAC)(
            curr_freq=cfg.algorithm.curriculum.freq,
            env=wrapped_env,
            **cfg.algorithm.model,
            policy_kwargs=dict(
                max_horizon=max_horizon,
                strategy="time",
                horizons=np.arange(max_horizon, -1, -max_horizon // n,)
            ),
        )

    if cfg.algorithm.name == "Residual_SAC":
        wrapped_env = ResidualWrapper(base_env)
        model = ResidualSAC(env=wrapped_env, **cfg.algorithm.model)

    else:  # Native SAC class is utilized to construct the model
        if cfg.algorithm.name == "SAC_Split":
            wrapped_env = SplitWrapper(base_env)
        else:
            wrapped_env = StandaloneWrapper(base_env)

        model = SAC(env=wrapped_env, **cfg.algorithm.model)

    assert cfg.algorithm not in ["Curriculum_SAC", "Residual_SAC", "SAC_Split" or "SAC"], "Overwritten algorithm not supported"

    # set up evaluation environment to be the same as the training environment
    eval_env = wrapped_env

    # set up logger
    tmp_path = cfg.dir
    new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # set up callbacks
    callbacks = [
        PerformanceLog(eval_env=eval_env, **cfg.algorithm.eval, cfg=cfg),
        Training_info()
    ]

    # train the model
    model.learn(**cfg.algorithm.learn, callback=callbacks)


if __name__ == "__main__":
    main()


