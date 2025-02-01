# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a cart-pole and interact with it.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/01_assets/run_articulation.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with an articulation.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import numpy as np
import omni.physics.tensors as physics
import omni.physx as _physx

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import JointAttachmentCfg
from omni.isaac.orbit.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.orbit.sim import SimulationContext

from pxr import Gf, Usd, UsdGeom, UsdPhysics

##
# Pre-defined configs
##
from omni.isaac.orbit.actuators import ImplicitActuatorCfg
from omni.isaac.orbit.assets import Rope, RopeCfg
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a robot in it
    origins = [[5.25, 2.25, 0.0], [-5.25, 5.25, 0.0], [5.25, -2.25, 0.0], [-5.25, -2.25, 0.0]]
    for i, origin in enumerate(origins):
        prim_utils.create_prim(f"/World/Origin{i}", "Xform", translation=origin)

    # Add rigid objects
    cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/cube_left",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=True,
                locked_rot_axis=7,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-5.25, -2.25, 0.1)),
    )
    cube_left = RigidObject(cfg)

    cfg = RigidObjectCfg(
        prim_path="/World/Origin.*/cube_right",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=True,
                locked_rot_axis=7,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(10.35, -2.25, 0.1)),
    )
    cube_right = RigidObject(cfg)

    # Articulation
    rope_cfg = RopeCfg(
        prim_path="/World/Origin.*/Rope",
        spawn=sim_utils.RopeShapeCfg(
            num_links=40,
            length=20.0,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        ),
        init_state=RopeCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        attachments=[
            JointAttachmentCfg(
                joint_type="fixed",
                prim_path=cube_left.cfg.prim_path + "/geometry/mesh",
                attached_link_idx=0,
                local_pos0=(0.0, 0.0, 0.0),
                local_pos1=(-0.2, 0.0, 0.0),
            ),
            JointAttachmentCfg(
                joint_type="fixed",
                prim_path=cube_right.cfg.prim_path + "/geometry/mesh",
                attached_link_idx=39,
                local_pos0=(-0.2, 0.0, 0.0),
                local_pos1=(0.0, 0.0, 0.0),
            ),
        ],
    )
    rope = Rope(cfg=rope_cfg)

    # return the scene information
    scene_entities = {"rope": rope, "cube_left": cube_left, "cube_right": cube_right}
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    rope: Rope = entities["rope"]
    cube_left: RigidObject = entities["cube_left"]
    cube_right: RigidObject = entities["cube_right"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            link_state = rope.data.default_link_state.clone()
            cube_left_state = cube_left.data.default_root_state.clone()
            cube_right_state = cube_right.data.default_root_state.clone()

            cube_left_state[:, :3] += origins
            cube_right_state[:, :3] += origins
            cube_left.write_root_state_to_sim(cube_left_state)
            cube_right.write_root_state_to_sim(cube_right_state)
            cube_left.write_root_velocity_to_sim(torch.zeros_like(cube_left.data.root_vel_w))
            cube_right.write_root_velocity_to_sim(torch.zeros_like(cube_right.data.root_vel_w))
            # # link_state[:, :3] += origins
            # rope.write_root_state_to_sim(link_state)
            # # set joint positions with some noise
            # joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            # joint_pos += torch.rand_like(joint_pos) * 0.1
            # robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            rope.reset()
            cube_left.reset()
            cube_right.reset()
            print("[INFO]: Resetting robot state...")
        # Apply random action
        # -- generate random joint efforts
        efforts = torch.randn_like(cube_left.data.root_vel_w) * 10.0
        efforts[:, 2:] = 0.0
        efforts[:, 0] *= 0.1
        # # -- apply action to the robot
        cube_left.write_root_velocity_to_sim(efforts)
        cube_right.write_root_velocity_to_sim(efforts)
        # # -- write data to sim
        cube_left.write_data_to_sim()
        cube_right.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Update buffers
        rope.update()
        cube_left.update(sim_dt)
        cube_right.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
    # Design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
