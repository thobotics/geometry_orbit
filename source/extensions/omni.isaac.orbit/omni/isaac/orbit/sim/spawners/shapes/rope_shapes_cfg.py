# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from omni.isaac.orbit.sim.spawners import materials
from omni.isaac.orbit.sim.spawners.spawner_cfg import RopeSpawnerCfg
from omni.isaac.orbit.utils import configclass

from . import rope_shapes


@configclass
class RopeShapeCfg(RopeSpawnerCfg):

    func: Callable = rope_shapes.spawn_rope

    size: tuple[float, float, float] = None
    """Size of the rope."""

    """Configuration parameters for a USD Geometry or Geom prim."""

    visual_material_path: str = "/rope/visual_material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """
    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """

    physics_material_path: str = "/rope/physics_material"
    """Path to the physics material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `physics_material` is not None.
    """
    physics_material: materials.PhysicsMaterialCfg | None = None
    """Physics material properties.

    Note:
        If None, then no physics material will be added.
    """

    num_links: int | None = MISSING
    """Number of segments to use for the rope object.

    Note:
        If None, then the primitive shape will be used.
    """

    length: float | None = MISSING
    """Length of the rope object."""

    radius: float | None = 0.025  # 0.1
    """Radius of the rope object."""

    inv_mass: float | None = 1.0
    """Inverse mass of the rope object."""

    contact_offset: float | None = 0.001
    """Contact offset of the rope object."""

    rest_offset: float | None = 0.0
    """Rest offset of the rope object."""

    cone_angle_limit: float | None = 110.0
    """Cone angle limit of the rope object."""

    rope_damping: float | None = 1e10  # 1e10
    """Damping of the rope object."""

    rope_stiffness: float | None = 1e4
    """Stiffness of the rope object."""
