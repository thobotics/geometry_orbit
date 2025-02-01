# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import DeformableObjectCfg
from omni.isaac.orbit.sensors import FrameTransformerCfg
from omni.isaac.orbit.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.orbit.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg
from omni.isaac.orbit.utils import configclass

from omni.isaac.orbit_tasks.manipulation.lift_deformable import mdp
from omni.isaac.orbit_tasks.manipulation.lift_deformable.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
# isort: off
from omni.isaac.orbit.markers.config import FRAME_MARKER_CFG
from omni.isaac.orbit_assets.franka import FRANKA_PANDA_CFG  # isort: skip


@configclass
class FrankaDeformableCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=[0.3, 0, 0.04], rot=[1, 0, 0, 0]),
            spawn=sim_utils.DeformableCuboidCfg(
                size=(0.06, 0.06, 0.06),
                voxel_count=8,
                deformable_props=DeformableBodyPropertiesCfg(
                    deformable_enabled=True,
                    kinematic_enabled=False,
                    collision_simplification=True,
                    simulation_hexahedral_resolution=5,
                    vertex_velocity_damping=0.0,
                    solver_position_iteration_count=20,
                    sleep_threshold=0.05,
                    sleep_damping=1.0,
                    self_collision=True,
                    self_collision_filter_distance=0.05,
                ),
                physics_material=sim_utils.DeformableBodyMaterialCfg(),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )


@configclass
class FrankaDeformableCubeLiftEnvCfg_PLAY(FrankaDeformableCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
