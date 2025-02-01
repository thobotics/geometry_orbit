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
from omni.isaac.orbit.assets import ClothObject, ClothObjectCfg
from omni.isaac.orbit.sim import SimulationContext


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg())
    # Set main camera
    SimulationContext.set_camera_view(eye=(3.0, 10.0, 6.0), target=(3.0, 0.0, 0.0))

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights-1
    cfg = sim_utils.SphereLightCfg(intensity=6000.0, color=(0.75, 0.75, 0.75), radius=5.0)
    cfg.func("/World/Light/greyLight", cfg, translation=(4.5, 3.5, 10.0))
    # Lights-2
    cfg = sim_utils.SphereLightCfg(intensity=6000.0, color=(1.0, 1.0, 1.0), radius=5.0)
    cfg.func("/World/Light/whiteSphere", cfg, translation=(-4.5, 3.5, 10.0))

    # Define cloth properties
    cloth_cfg = schemas.ClothPropertiesCfg(
        spring_stretch_stiffness=1e4,
        spring_bend_stiffness=200.0,
        spring_shear_stiffness=100.0,
        spring_damping=0.2,
    )

    particle_material_cfg = sim_utils.ParticleMaterialCfg(drag=0.1, lift=0.3, friction=0.6)

    radius = 0.5 * (0.6 / 5.0)
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
    distance_between_assets = 2.0
    num_instances = 4
    for i in range(num_instances):
        x = i * distance_between_assets
        prim_utils.create_prim(f"/World/Objects{i}", "Xform", translation=(x, 0.0, 0.0))

    spawn_fnc = sim_utils.ParticleClothCfg(
        size=(25, 25),
        cloth_props=cloth_cfg,
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        particle_material=particle_material_cfg,
        particle_system_props=particle_system,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
    )

    # Setup deformable object
    cfg = ClothObjectCfg(
        prim_path="/World/Objects.*/plain_cloth",
        spawn=spawn_fnc,
        init_state=ClothObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 2.0),
            rot=(0.9238795042037964, 0.0, 0.3826834261417389, 0.0),
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
            points_state[:, :, :3] = initial_points_pos + torch.tensor([0.0, 0.0, 2.0], device="cuda:0")
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
