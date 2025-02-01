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

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a deformable kinematic environment.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import traceback

import carb

import omni.isaac.orbit.envs.mdp as mdp
import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.sim.schemas as schemas
from omni.isaac.orbit.assets import (
    AssetBaseCfg,
    BodyAttachmentCfg,
    DeformableObjectCfg,
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
        return 0.15

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


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Example scene configuration.

    The scene comprises of a light source and a deformable mesh controlled by to two attached cubes.
    """

    # add cube
    # cube_left: FixedObjectCfg = FixedObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/cube_left",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.8, 0.8, 0.8),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.5, 0.0, 7.5)),
    # )

    cube_left: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_left",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-2.5, 0.0, 7.5)),
    )

    # cube_right: FixedObjectCfg = FixedObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/cube_right",
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.8, 0.8, 0.8),
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, 0.0, 7.5)),
    # )

    cube_right: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_right",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True, kinematic_enabled=True, disable_gravity=True
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.5, 0.0, 7.5)),
    )

    cube_bottom: FixedObjectCfg = FixedObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube_bottom",
        spawn=sim_utils.CuboidCfg(
            size=(5.0, 0.8, 0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.5)),
    )

    deformable_mesh = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/deformable_mesh",
        attachments=[
            BodyAttachmentCfg(
                prim_path="{ENV_REGEX_NS}/cube_left",
                name="attachment_left",
            ),
            BodyAttachmentCfg(
                prim_path="{ENV_REGEX_NS}/cube_right",
                name="attachment_right",
            ),
            BodyAttachmentCfg(
                prim_path="{ENV_REGEX_NS}/cube_bottom",
                name="attachment_bottom",
            ),
        ],
        spawn=sim_utils.RopeShapeCfg(
            size=(5.0, 0.1, 5.0),
            segment_count=16,
            deformable_props=schemas.DeformableBodyPropertiesCfg(
                simulation_hexahedral_resolution=6,
                collision_simplification=True,
                kinematic_enabled=False,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                youngs_modulus=5e6,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0), metallic=0.2),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.5)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# Environment settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    left_joint_pos = CubeActionTermCfg(asset_name="cube_left")
    right_joint_pos = CubeActionTermCfg(asset_name="cube_right")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # cube velocity
        left_position = ObsTerm(func=base_position, params={"asset_cfg": SceneEntityCfg("cube_left")})
        right_position = ObsTerm(func=base_position, params={"asset_cfg": SceneEntityCfg("cube_right")})
        nodal_points = ObsTerm(
            func=mdp.deformable_nodal_points,
            params={"asset_cfg": SceneEntityCfg("deformable_mesh")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    reset_cube_left = RandTerm(
        func=mdp.reset_fixed_body_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "z": (-0.5, 0.5)},
            "asset_cfg": SceneEntityCfg("cube_left"),
        },
    )

    reset_cube_right = RandTerm(
        func=mdp.reset_fixed_body_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "z": (-0.5, 0.5)},
            "asset_cfg": SceneEntityCfg("cube_right"),
        },
    )


##
# Environment configuration
##


@configclass
class DeformableKinematicEnvCfg(BaseEnvCfg):
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
        self.decimation = 2
        # simulation settings
        self.sim.dt = 0.01

        # Note: the following settings are copied from OIGE FrankaDeformable
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_correlation_distance = 0.025
        self.sim.physx.gpu_max_rigid_contact_count = 512 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 32 * 1024 * 1024
        self.sim.physx.gpu_found_lost_pairs_capacity = 512 * 1024
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 256 * 1024
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024
        self.sim.physx.gpu_max_soft_body_contacts = 4 * 1024 * 1024
        self.sim.physx.gpu_max_particle_contacts = 1024 * 1024
        self.sim.physx.gpu_heap_capacity = 32 * 1024 * 1024
        self.sim.physx.gpu_temp_buffer_capacity = 16 * 1024 * 1024
        self.sim.physx.gpu_max_num_partitions = 8


def main():
    """Main function."""

    # setup base environment
    env = BaseEnv(cfg=DeformableKinematicEnvCfg())

    # setup target position commands
    velocity_command = torch.rand(env.num_envs, 6, device=env.device) * 0.01

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
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
