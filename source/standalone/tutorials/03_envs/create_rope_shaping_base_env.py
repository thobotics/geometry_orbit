# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script creates a simple deformable kinematic environment with two cubes and a deformable mesh.
The cubes are kinematic and can be controlled by the user. The deformable mesh is attached to the cubes
and is simulated using the FEM simulator.

.. code-block:: bash

    # Run the script
    ./orbit.sh -p source/standalone/tutorials/03_envs/create_deformable_kinematic_base_env.py --num_envs 32
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse
import enum
import os

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a deformable kinematic environment.")
parser.add_argument("--num_envs", type=int, default=40, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch the simulator
# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
else:
    app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.kit"

# launch omniverse app
app_launcher = AppLauncher(args_cli, experience=app_experience)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import traceback
from collections.abc import Sequence

import carb

import omni.isaac.orbit.envs.mdp as mdp
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.sim.schemas as schemas
from omni.isaac.orbit.assets import (
    AssetBaseCfg,
    RopeCfg,
    RigidObjectCfg,
    JointAttachmentCfg,
    FixedObject,
    RigidObject,
)
from omni.isaac.orbit.envs import BaseEnv, BaseEnvCfg
from omni.isaac.orbit.managers import ActionTerm, ActionTermCfg
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.timer import Timer

##
# Custom action term
##


class CubeActionTerm(ActionTerm):
    """Simple action term that apply a velocity command to the cube."""

    _asset: FixedObject | RigidObject
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: CubeActionTermCfg, env: BaseEnv):
        # call super constructor
        super().__init__(cfg, env)
        # create buffers
        self._raw_actions = torch.zeros(env.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=self.device)

    """
    Properties.
    """

    @property
    def action_scale(self) -> float:
        return 5.0

    @property
    def action_max(self) -> float:
        return 1.0

    @property
    def action_dim(self) -> int:
        return self._raw_actions.shape[1]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    """
    Operations
    """

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # no-processing of actions
        self._processed_actions[:] = self._raw_actions[:]

    def apply_actions(self):
        vel_command = self._processed_actions
        vel_command = torch.clamp(
            vel_command,
            -self.action_max,
            self.action_max,
        )
        vel_command = vel_command * self.action_scale

        if isinstance(self._asset, RigidObject):
            # add zero angular velocity
            self._asset.write_root_velocity_to_sim(torch.cat([vel_command, torch.zeros_like(vel_command)], dim=-1))
        else:
            vel_command *= self._env.physics_dt
            current_pose = self._asset.data.root_state_w
            current_pose[:, :3] = current_pose[:, :3] + vel_command
            self._asset.write_root_pose_to_sim(current_pose)


@configclass
class CubeActionTermCfg(ActionTermCfg):
    """Configuration for the cube action term."""

    class_type: type = CubeActionTerm
    """The class corresponding to the action term."""


##
# Custom observation term
##


def base_position(env: BaseEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: FixedObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


##
# Scene definition
##
NUM_LINKS = 40
LENGTH = 5.0


class GRIPPER_TYPE(enum.Enum):
    """Gripper type."""

    CUBOID = enum.auto()


GripperType = GRIPPER_TYPE.CUBOID


class Grippers:
    N_GRIPPERS = 3

    # add cube
    grippers_init_state = [
        (0.0, 0.0, 0.1),
        (3.9, 0.0, 0.1),  # =(LENGTH / NUM_LINKS - radius) * (NUM_LINKS - 1)
        (1.95, 0.0, 0.1),  # =(LENGTH / NUM_LINKS - radius) * (NUM_LINKS - 1) / 2
    ]

    local_pos0 = [
        (0.0, 0.0, 0.0),
        (-0.05, 0.0, 0.0),
        (0.0, -0.1, 0.0),
    ]

    local_pos1 = [
        (-0.05, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]

    grippers_link_indices = [0, NUM_LINKS - 1, NUM_LINKS // 2]

    @classmethod
    def default_cfg(cls, rigid: bool = True, **kwargs):
        return cls.cuboid_cfg(rigid=rigid, **kwargs)

    @classmethod
    def default_attachment_path(cls, rigid: bool = True):
        return "/geometry/mesh"

    @classmethod
    def build_physics_props(cls, rigid: bool = True):
        if rigid:
            return {
                "rigid_props": sim_utils.RigidBodyPropertiesCfg(
                    rigid_body_enabled=True,
                    kinematic_enabled=False,
                    disable_gravity=True,
                    linear_damping=10.0,  # high damping to reduce oscillations
                    locked_rot_axis=7,
                ),
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            }
        else:
            return {
                "collision_props": sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            }

    @classmethod
    def cuboid_cfg(cls, rigid: bool = True, size: tuple[float, float, float] = (0.1, 0.1, 0.1)):
        physics_props = cls.build_physics_props(rigid)
        return sim_utils.CuboidCfg(
            size=size,
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
            **physics_props,
        )


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration.

    The scene comprises of a light source and a deformable mesh controlled by to two attached cubes.
    """

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(1e4, 1e4), color=(0.0, 0.0, 0.01)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.5)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.SphereLightCfg(color=(0.75, 0.75, 0.75), intensity=10000.0, radius=5.0),
    )

    def __post_init__(self):
        """Post initialization."""
        spawn_gripper = Grippers.default_cfg(rigid=True)
        for i in range(Grippers.N_GRIPPERS):
            self.__dict__[f"cube_{i}"] = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/" + f"cube_{i}",
                spawn=spawn_gripper,
                init_state=RigidObjectCfg.InitialStateCfg(pos=Grippers.grippers_init_state[i]),
            )

        # This make sure grippers are added before cloth
        self.__dict__["rope"] = RopeCfg(
            prim_path="{ENV_REGEX_NS}/rope",
            attachments=[
                JointAttachmentCfg(
                    joint_type="fixed",
                    prim_path="{ENV_REGEX_NS}/" + f"cube_{i}" + Grippers.default_attachment_path(),
                    attached_link_idx=Grippers.grippers_link_indices[i],
                    local_pos0=Grippers.local_pos0[i],
                    local_pos1=Grippers.local_pos1[i],
                    name=f"attachment_{i}",
                )
                for i in range(Grippers.N_GRIPPERS)
            ],
            spawn=sim_utils.RopeShapeCfg(
                num_links=NUM_LINKS,
                length=LENGTH,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RopeCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        )


##
# Environment settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    def __post_init__(self):
        """Post initialization."""
        for i in range(Grippers.N_GRIPPERS):
            self.__dict__[f"cube_{i}"] = CubeActionTermCfg(asset_name=f"cube_{i}")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # links_positions = ObsTerm(
        #     func=mdp.links_positions,
        #     params={
        #         "asset_cfg": SceneEntityCfg("rope"),
        #     },
        # )

        def __post_init__(self):
            """Post initialization."""
            for i in range(Grippers.N_GRIPPERS):
                self.__dict__[f"cube_{i}"] = ObsTerm(
                    func=base_position,
                    params={"asset_cfg": SceneEntityCfg(f"cube_{i}")},
                )

            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    def __post_init__(self):
        """Post initialization."""
        for i in range(Grippers.N_GRIPPERS):
            self.__dict__[f"reset_cube_{i}"] = RandTerm(
                func=mdp.reset_fixed_body_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.1, 0.1), "z": (-0.1, 0.1)},
                    "asset_cfg": SceneEntityCfg(f"cube_{i}"),
                },
            )


##
# Environment configuration
##


@configclass
class RopeShapingEnvCfg(BaseEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=10.0, replicate_physics=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    randomization: RandomizationCfg = RandomizationCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 1
        # simulation settings
        self.sim.dt = 0.01

        # Note: the following settings are copied from OIGE FrankaDeformable
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.025
        self.sim.physx.gpu_max_rigid_contact_count = 32 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 4 * 1024 * 1024
        self.sim.physx.gpu_found_lost_pairs_capacity = 32 * 1024
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.sim.physx.gpu_max_soft_body_contacts = 1 * 256 * 1024
        self.sim.physx.gpu_max_particle_contacts = 256 * 1024
        self.sim.physx.gpu_heap_capacity = 2 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 1 * 1024 * 1024
        self.sim.physx.gpu_max_num_partitions = 2


def main():
    """Main function."""

    # setup base environment
    env = BaseEnv(cfg=RopeShapingEnvCfg())
    SimulationContext.set_camera_view(eye=(0.0, 20.0, 30.0), target=(3.0, 0.0, 5.0))

    # setup target position commands
    # velocity_command = (
    #     torch.rand(env.num_envs, 3 * Grippers.N_GRIPPERS, device=env.device) * 0.1
    # )
    velocity_command = torch.zeros(env.num_envs, Grippers.N_GRIPPERS, 3, device=env.device)
    velocity_command[..., 1] -= 0.4
    velocity_command = velocity_command.view(env.num_envs, -1)

    # simulate physics
    count = 0

    for _ in range(10):
        obs, _ = env.step(torch.zeros_like(velocity_command))

    obs, _ = env.reset()

    timer = Timer()
    timer.start()
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
                timer.stop()
                print(f"[INFO]: Time elapsed: {timer.total_run_time}")
                timer.start()
            # step env
            obs, _ = env.step(velocity_command)
            # update counter
            count += 1

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
