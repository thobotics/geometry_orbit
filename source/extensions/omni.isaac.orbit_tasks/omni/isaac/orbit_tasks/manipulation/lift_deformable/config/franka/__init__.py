# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import gymnasium as gym

from . import agents, env_cfg

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Lift-Deformable-Cube-Franka-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg.FrankaDeformableCubeLiftEnvCfg,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftCubePPORunnerCfg,
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Deformable-Cube-Franka-Play-v0",
    entry_point="omni.isaac.orbit.envs:RLTaskEnv",
    kwargs={
        "env_cfg_entry_point": env_cfg.FrankaDeformableCubeLiftEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.LiftCubePPORunnerCfg,
    },
    disable_env_checker=True,
)
