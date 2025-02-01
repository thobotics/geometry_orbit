# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to work with the deformable-body objects in Orbit.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/01_assets/deformable_objects.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Deformable-body objects in Orbit.")
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
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.sim.schemas as schemas
from omni.isaac.orbit.assets import DeformableObject, DeformableObjectCfg
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg())
    # Set main camera
    sim.set_camera_view(eye=[7.0, 7.0, 7.0], target=[3.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights-1
    cfg = sim_utils.SphereLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75), radius=2.5)
    cfg.func("/World/Light/greyLight", cfg, translation=(4.5, 3.5, 10.0))
    # Lights-2
    cfg = sim_utils.SphereLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0), radius=2.5)
    cfg.func("/World/Light/whiteSphere", cfg, translation=(-4.5, 3.5, 10.0))

    # Define deformable body properties
    deformable_cfg = schemas.DeformableBodyPropertiesCfg(
        vertex_velocity_damping=0.0,
        sleep_damping=1.0,
        sleep_threshold=0.05,
        settling_threshold=0.1,
        self_collision=True,
        self_collision_filter_distance=0.05,
        solver_position_iteration_count=20,
        kinematic_enabled=False,
        simulation_hexahedral_resolution=2,
        collision_simplification=True,
    )
    deformable_material_cfg = sim_utils.DeformableBodyMaterialCfg(
        dynamic_friction=0.5,
        youngs_modulus=5e4,
        poissons_ratio=0.4,
        damping_scale=0.1,
        elasticity_damping=0.1,
    )

    # Add objects
    distance_between_assets = 2.0
    num_instances = 4
    for i in range(num_instances):
        x = i * distance_between_assets
        prim_utils.create_prim(f"/World/Objects{i}", "Xform", translation=(x, 0.0, 0.0))

    spawn_type = "voxel"  # "usd" or "voxel"
    if spawn_type == "usd":
        spawn_fnc = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cube.usd",
            deformable_props=deformable_cfg,
            physics_material=deformable_material_cfg,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        )
    else:
        spawn_fnc = sim_utils.DeformableCuboidCfg(
            size=(1.0, 1.0, 1.0),
            voxel_count=8,
            deformable_props=deformable_cfg,
            physics_material=deformable_material_cfg,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
        )

    # Setup deformable object
    cfg = DeformableObjectCfg(
        prim_path="/World/Objects.*/Cube",
        spawn=spawn_fnc,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )
    # Create deformable object handler
    deformable_object = DeformableObject(cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    count = 0

    # update buffers
    deformable_object.update()
    initial_nodal_pos = deformable_object.data.nodal_pos_w.clone()

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 250 == 0:
            # reset counter
            count = 0
            # reset root state
            root_state = deformable_object.data.default_nodal_state_w.clone()
            # -- update position
            root_state[:, : root_state.size(1) // 2, :] = initial_nodal_pos + torch.tensor(
                [0.0, 0.0, 2.0], device="cuda:0"
            )
            # -- set root state
            deformable_object.write_root_state_to_sim(root_state)
            # reset buffers
            deformable_object.reset()
            print(">>>>>>>> Reset!")
        # perform step
        sim.step()
        # update counter
        count += 1
        # update buffers
        deformable_object.update()


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
