# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to work with the cloth objects in Orbit.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/01_assets/run_cloths.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Cloth objects in Orbit.")
parser.add_argument("--usd", type=str, default="", help="Path to the usd file.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.core.utils.prims as prim_utils

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.sim.schemas as schemas
from omni.isaac.orbit.assets import (
    BodyAttachmentCfg,
    ClothObject,
    ClothObjectCfg,
    FixedObject,
    FixedObjectCfg,
    RigidObjectCfg,
    RigidObject,
)
from omni.isaac.orbit.sim import SimulationContext


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg())
    # Set main camera
    SimulationContext.set_camera_view(eye=(3.0, 7.0, 6.0), target=(3.0, 0.0, 5.0))

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights-1
    cfg = sim_utils.SphereLightCfg(
        intensity=6000.0, color=(0.75, 0.75, 0.75), radius=5.0
    )
    cfg.func("/World/Light/greyLight", cfg, translation=(4.5, 3.5, 10.0))
    # Lights-2
    cfg = sim_utils.SphereLightCfg(intensity=6000.0, color=(1.0, 1.0, 1.0), radius=5.0)
    cfg.func("/World/Light/whiteSphere", cfg, translation=(-4.5, 3.5, 10.0))

    # Define cloth properties
    cloth_cfg = schemas.ClothPropertiesCfg(
        spring_stretch_stiffness=1e6,
        spring_bend_stiffness=1.0,
        spring_shear_stiffness=100.0,
        spring_damping=0.02,
        cloth_path="mesh",
    )

    particle_material_cfg = sim_utils.ParticleMaterialCfg(drag=0.1, friction=0.2)

    radius = 0.25 * (0.6 / 5.0)
    restOffset = radius
    contactOffset = restOffset * 1.5
    particle_system = schemas.ParticleSystemPropertiesCfg(
        rest_offset=restOffset,
        contact_offset=contactOffset,
        solid_rest_offset=restOffset,
        fluid_rest_offset=restOffset,
        particle_contact_offset=contactOffset,
    )

    # Add objects
    distance_between_assets = 5.0
    num_instances = 4
    for i in range(num_instances):
        x = i * distance_between_assets
        prim_utils.create_prim(f"/World/Objects{i}", "Xform", translation=(x, 0.0, 0.0))

    # Add rigid objects
    cfg = FixedObjectCfg(
        prim_path="/World/Objects.*/cube_left",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=FixedObjectCfg.InitialStateCfg(pos=(-0.5, 0.0, 5.0)),
    )
    cube_left = FixedObject(cfg)

    cfg = FixedObjectCfg(
        prim_path="/World/Objects.*/cube_right",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=FixedObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 5.0)),
    )
    cube_right = FixedObject(cfg)

    cfg = RigidObjectCfg(
        prim_path="/World/Objects.*/hanger",
        spawn=sim_utils.CuboidCfg(
            size=(0.15, 1.0, 0.15),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 4.5)),
    )
    hanger = RigidObject(cfg)

    usd_path = args_cli.usd
    # usd_path = "/workspace/orbit/source/assets/TNNC_model2_063.usda"

    if not usd_path:
        spawn_fnc = sim_utils.SquareClothWithHoles(
            size=(15, 15),
            holes=[(7.5, 7.5, 1.0)],
            cloth_props=cloth_cfg,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            particle_material=particle_material_cfg,
            particle_system_props=particle_system,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), metallic=0.2
            ),
        )
    else:
        spawn_fnc = sim_utils.ClothUsdFileCfg(
            usd_path=usd_path,
            scale=(0.04, 0.04, 0.04),
            cloth_props=cloth_cfg,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            particle_material=particle_material_cfg,
            particle_system_props=particle_system,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.8, 0.8, 0.8), metallic=0.2
            ),
        )

    # Setup deformable object
    cfg = ClothObjectCfg(
        prim_path="/World/Objects.*/plain_cloth",
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
        init_state=ClothObjectCfg.InitialStateCfg(
            # pos=(0.0, -5.6, 5.1),
            pos=(0.0, 0.0, 4.5),
            rot=(0.707, 0.707, 0.0, 0.0),
        ),
    )
    # Create deformable object handler
    cloth_object = ClothObject(cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    count = 0

    # update buffers
    cloth_object.update()
    initial_points_pos = cloth_object.data.points_pos_w.clone()

    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 100 == 0:
            # reset counter
            count = 0
            # reset root state
            points_state = cloth_object.data.default_points_state_w.clone()
            points_state[:, :, :3] = initial_points_pos + torch.tensor(
                [0.0, 0.0, 2.0], device="cuda:0"
            )
            # -- set root state
            cloth_object.write_root_state_to_sim(points_state)
            # reset buffers
            cloth_object.reset()
            print(">>>>>>>> Reset!")
        # perform step
        sim.step()
        # update counter
        count += 1
        # update buffers
        cloth_object.update()


if __name__ == "__main__":
    main()
    # close sim app
    simulation_app.close()
