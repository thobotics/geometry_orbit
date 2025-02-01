# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import traceback
import unittest

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.sim.schemas as schemas
from omni.isaac.orbit.assets import ClothObject, ClothObjectCfg


class TestClothObject(unittest.TestCase):
    """Test for cloth object class."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # Simulation time-step
        self.dt = 0.01
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=self.dt, device="cuda:0")
        self.sim = sim_utils.SimulationContext(sim_cfg)

    def tearDown(self):
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        # clear the stage
        self.sim.clear_instance()

    """
    Tests
    """

    def test_initialization(self):
        """Test initialization for with rigid body API at the provided prim path."""
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
        num_instances = 1
        for i in range(num_instances):
            x = i * distance_between_assets
            prim_utils.create_prim(f"/World/Objects{i}", "Xform", translation=(x, 0.0, 0.0))

        spawn_fnc = sim_utils.ParticleClothCfg(
            size=(5, 5),
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

        # Play sim
        self.sim.reset()
        # Check if object is initialized
        self.assertTrue(cloth_object._is_initialized)
        self.assertEqual(len(cloth_object.body_names), 1)
        # Check buffers that exists and have correct shapes
        self.assertTrue(cloth_object.data.root_pos_w.shape == (1, 3))
        self.assertTrue(cloth_object.data.root_quat_w.shape == (1, 4))
        self.assertTrue(cloth_object.data.points_state_w.shape[2] == 6)

        # Simulate physics
        for _ in range(20):
            # perform rendering
            self.sim.step()
            # update object
            cloth_object.update()


if __name__ == "__main__":
    try:
        unittest.main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
