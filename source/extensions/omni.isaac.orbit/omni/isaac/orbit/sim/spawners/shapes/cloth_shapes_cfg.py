# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

import omni.isaac.orbit.sim.schemas as schemas
import omni.isaac.orbit.sim.spawners.materials as materials
from omni.isaac.orbit.sim.spawners.spawner_cfg import ClothObjectSpawnerCfg
from omni.isaac.orbit.utils import configclass

from .cloth_shapes import spawn_cloth_with_holes, spawn_plain_cloth

"""
Spawning cloth.
"""


@configclass
class ParticleClothCfg(ClothObjectSpawnerCfg):
    """Spawn a particle cloth prim.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """

    func: Callable = spawn_plain_cloth

    size: tuple[int, int] | None = None
    """Size of the asset. Defaults to None, which means the size will be set to (5, 5)."""

    mass_props: schemas.MassPropertiesCfg | None = None

    particle_material_path: str = "particle_material"
    """Path to the particle material to use for the prim. Defaults to "particle_material"."""

    particle_material: materials.ParticleMaterialCfg | None = None
    """Cloth material properties to override the cloth material properties in the USD file.
    Note:
        If None, then no cloth material will be added.
    """

    particle_system_props: schemas.ParticleSystemPropertiesCfg | None = None
    """Properties to apply to the particle system."""

    visual_material_path: str = "visual_material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """

    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties to override the visual material properties in the URDF file.

    Note:
        If None, then no visual material will be added.
    """


@configclass
class SquareClothWithHoles(ParticleClothCfg):
    """Spawn a square cloth with holes.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """

    func: Callable = spawn_cloth_with_holes

    holes: list[tuple[int, int, float]] = []
