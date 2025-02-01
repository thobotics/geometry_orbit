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
parser = argparse.ArgumentParser(description="Deformable-body objects with attachments in Orbit.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import traceback

import carb
import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.sim.schemas as schemas
from omni.isaac.orbit.assets import (
    BodyAttachmentCfg,
    DeformableObject,
    DeformableObjectCfg,
    FixedObject,
    FixedObjectCfg,
    RigidObject,
    RigidObjectCfg,
)
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    """Main function."""

    physx_config = sim_utils.PhysxCfg(
        gpu_max_rigid_contact_count=128 * 1024,
        gpu_max_rigid_patch_count=8 * 1024 * 1024,
        gpu_found_lost_pairs_capacity=128 * 1024,
        gpu_found_lost_aggregate_pairs_capacity=128 * 1024,
        gpu_total_aggregate_pairs_capacity=128 * 1024,
        gpu_max_soft_body_contacts=1 * 1024 * 1024,
        gpu_max_particle_contacts=128 * 1024,
        gpu_heap_capacity=8 * 1024 * 1024,
        gpu_temp_buffer_capacity=4 * 1024 * 1024,
        gpu_max_num_partitions=4,
    )

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(physx=physx_config))
    # Set main camera
    sim.set_camera_view(eye=[30.0, 20.0, 11.0], target=[0.0, 0.0, 0.0])

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights-1
    cfg = sim_utils.SphereLightCfg(intensity=6000.0, color=(0.75, 0.75, 0.75), radius=2.5)
    cfg.func("/World/Light/greyLight", cfg, translation=(4.5, 3.5, 10.0))
    # Lights-2
    cfg = sim_utils.SphereLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0), radius=2.5)
    cfg.func("/World/Light/whiteSphere", cfg, translation=(-4.5, 3.5, 10.0))

    # Define deformable body properties
    deformable_cfg = schemas.DeformableBodyPropertiesCfg(
        simulation_hexahedral_resolution=20,
        collision_simplification=True,
        kinematic_enabled=False,
    )
    deformable_material_cfg = sim_utils.DeformableBodyMaterialCfg(
        youngs_modulus=5e6,
    )

    # Add objects
    distance_between_assets = 15.0
    num_instances = 4
    for i in range(num_instances):
        x = i * distance_between_assets
        prim_utils.create_prim(f"/World/Objects{i}", "Xform", translation=(x, 0.0, 0.0))

    # Add rigid objects
    cfg = FixedObjectCfg(
        prim_path="/World/Objects.*/cube_left",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-3.5, 0.0, 7.5)),
    )
    cube_left = FixedObject(cfg)

    cfg = FixedObjectCfg(
        prim_path="/World/Objects.*/cube_right",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(3.5, 0.0, 7.5)),
    )
    cube_right = FixedObject(cfg)

    cfg = FixedObjectCfg(
        prim_path="/World/Objects.*/cube_bottom",
        spawn=sim_utils.CuboidCfg(
            size=(7.5, 0.8, 0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.5)),
    )
    cube_bottom = FixedObject(cfg)

    # Setup deformable object
    spawn_type = "voxel"  # "usd" or "voxel"
    if spawn_type == "usd":
        spawn_fnc = sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cube.usd",
            deformable_props=deformable_cfg,
            physics_material=deformable_material_cfg,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0), metallic=0.2),
        )
    else:
        spawn_fnc = sim_utils.DeformableTrapezoidCfg(
            size=(7.5, 0.1, 5.0),
            top_width_ratio=1.2,
            base_width_ratio=0.5,
            voxel_count=64,
            deformable_props=deformable_cfg,
            physics_material=deformable_material_cfg,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 0.0), metallic=0.2),
        )

    cfg = DeformableObjectCfg(
        prim_path="/World/Objects.*/deformable_mesh",
        attachments=[
            BodyAttachmentCfg(
                prim_path="/World/Objects.*/cube_left",
                name="attachment_left",
            ),
            BodyAttachmentCfg(
                prim_path="/World/Objects.*/cube_right",
                name="attachment_right",
            ),
            BodyAttachmentCfg(
                prim_path="/World/Objects.*/cube_bottom",
                name="attachment_bottom",
            ),
        ],
        spawn=spawn_fnc,
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.5)),
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
            root_state[:, : root_state.size(1) // 2, :] = initial_nodal_pos
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
