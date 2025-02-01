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
import os

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a deformable kinematic environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")

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
    BodyAttachmentCfg,
    ClothObject,
    ClothObjectCfg,
    FixedObject,
    FixedObjectCfg,
    RigidObject,
    RigidObjectCfg,
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
        return 0.4

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

        if isinstance(self._asset, RigidObject):
            # add zero angular velocity
            self._asset.write_root_velocity_to_sim(torch.cat([vel_command, torch.zeros_like(vel_command)], dim=-1))
        else:
            vel_command = torch.clamp(
                vel_command * self._env.physics_dt * self.action_scale,
                -self.action_max,
                self.action_max,
            )
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


class Grippers:

    N_GRIPPERS = 4

    # add cube
    grippers_init_state = [
        (-0.5, 0.0, 5.0),
        (0.5, 0.0, 5.0),
        (-0.5, 0.0, 4.0),
        (0.5, 0.0, 4.0),
        # (-0.5, 0.0, 4.5),
        # (0.5, 0.0, 4.5),
        # (0.0, 0.0, 5.0),
        # (0.0, 0.0, 4.0),
    ]

    @classmethod
    def default_cfg(cls, rigid: bool = True):
        if rigid:
            return [
                RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + f"cube_{i}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.1, 0.1, 0.1),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(
                            rigid_body_enabled=True,
                            kinematic_enabled=False,
                            disable_gravity=True,
                            linear_damping=10.0,  # high damping to reduce oscillations
                            locked_rot_axis=7,
                        ),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                        physics_material=sim_utils.RigidBodyMaterialCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=cls.grippers_init_state[i]),
                )
                for i in range(cls.N_GRIPPERS)
            ]
        else:
            return [
                FixedObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + f"cube_{i}",
                    spawn=sim_utils.CuboidCfg(
                        size=(0.1, 0.1, 0.1),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                        physics_material=sim_utils.RigidBodyMaterialCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
                    ),
                    init_state=FixedObjectCfg.InitialStateCfg(pos=cls.grippers_init_state[i]),
                )
                for i in range(cls.N_GRIPPERS)
            ]


num_particles_per_row = 15
cloth_size = (num_particles_per_row, num_particles_per_row)
cloth_holes = [(
    num_particles_per_row / 2,
    num_particles_per_row / 2,
    num_particles_per_row / 15,
)]

radius = 0.5 * 1 / (num_particles_per_row + 1)
restOffset = radius
contactOffset = restOffset * 1.5


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration.

    The scene comprises of a light source and a deformable mesh controlled by to two attached cubes.
    """

    hanger = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/hanger",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/workspace/ironlib/ironlib/orbit/tasks/manipulation/assets/cylinder.usd",
            scale=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=True,
                locked_pos_axis=7,
                locked_rot_axis=7,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.3)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.9, 4.5)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.SphereLightCfg(color=(0.75, 0.75, 0.75), intensity=10000.0, radius=5.0),
    )

    def __post_init__(self):
        """Post initialization."""
        grippers = Grippers.default_cfg(rigid=True)
        for i in range(Grippers.N_GRIPPERS):
            self.__dict__[f"cube_{i}"] = grippers[i]

        # This make sure grippers are added before cloth
        self.__dict__["cloth"] = ClothObjectCfg(
            prim_path="{ENV_REGEX_NS}/plain_cloth",
            attachments=[
                BodyAttachmentCfg(
                    prim_path="{ENV_REGEX_NS}/" + f"cube_{i}",
                    name=f"attachment_{i}",
                )
                for i in range(Grippers.N_GRIPPERS)
            ],
            spawn=sim_utils.SquareClothWithHoles(
                size=cloth_size,
                holes=cloth_holes,
                cloth_props=schemas.ClothPropertiesCfg(
                    spring_stretch_stiffness=2e6,
                    spring_bend_stiffness=1.0,
                    spring_shear_stiffness=100.0,
                    spring_damping=0.02,
                    cloth_path="mesh",
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                particle_material=sim_utils.ParticleMaterialCfg(drag=0.1, friction=0.2),
                particle_system_props=schemas.ParticleSystemPropertiesCfg(
                    rest_offset=restOffset,
                    contact_offset=contactOffset,
                    solid_rest_offset=restOffset,
                    fluid_rest_offset=restOffset,
                    particle_contact_offset=contactOffset,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=ClothObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 4.5),
                rot=(0.707, 0.707, 0.0, 0.0),
            ),
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

        points_state = ObsTerm(
            func=mdp.cloth_points_state,
            params={"asset_cfg": SceneEntityCfg("cloth")},
        )

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
class ClothInsertionEnvCfg(BaseEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0, replicate_physics=False)
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
        self.sim.physx.gpu_max_rigid_contact_count = 128 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 8 * 1024 * 1024
        self.sim.physx.gpu_found_lost_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 64 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 128 * 1024
        self.sim.physx.gpu_max_soft_body_contacts = 1 * 1024 * 1024
        self.sim.physx.gpu_max_particle_contacts = 1024 * 1024
        self.sim.physx.gpu_heap_capacity = 8 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 4 * 1024 * 1024
        self.sim.physx.gpu_max_num_partitions = 2


def main():
    """Main function."""

    # setup base environment
    env = BaseEnv(cfg=ClothInsertionEnvCfg())
    SimulationContext.set_camera_view(eye=(3.0, 7.0, 6.0), target=(3.0, 0.0, 5.0))

    # setup target position commands
    # velocity_command = (
    #     torch.rand(env.num_envs, 3 * Grippers.N_GRIPPERS, device=env.device) * 0.1
    # )
    velocity_command = torch.zeros(env.num_envs, Grippers.N_GRIPPERS, 3, device=env.device)
    velocity_command[..., 1] -= 0.4
    velocity_command = velocity_command.view(env.num_envs, -1)

    # simulate physics
    count = 0
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
