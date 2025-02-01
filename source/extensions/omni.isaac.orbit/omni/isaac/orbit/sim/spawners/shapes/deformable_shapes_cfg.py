# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from omni.isaac.orbit.sim.spawners import materials
from omni.isaac.orbit.sim.spawners.spawner_cfg import DeformableObjectSpawnerCfg
from omni.isaac.orbit.utils import configclass

from . import deformable_shapes


@configclass
class DeformableShapeCfg(DeformableObjectSpawnerCfg):
    """Configuration parameters for a USD Geometry or Geom prim."""

    visual_material_path: str = "material"
    """Path to the visual material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `visual_material` is not None.
    """
    visual_material: materials.VisualMaterialCfg | None = None
    """Visual material properties.

    Note:
        If None, then no visual material will be added.
    """

    physics_material_path: str = "material"
    """Path to the physics material to use for the prim. Defaults to "material".

    If the path is relative, then it will be relative to the prim's path.
    This parameter is ignored if `physics_material` is not None.
    """
    physics_material: materials.PhysicsMaterialCfg | None = None
    """Physics material properties.

    Note:
        If None, then no physics material will be added.
    """

    voxel_count: int | None = None
    """Number of voxels to use for the deformable object.

    Note:
        If None, then the primitive shape will be used.
    """


@configclass
class DeformableCuboidCfg(DeformableShapeCfg):
    """Configuration parameters for a cuboid prim.

    See :meth:`spawn_deformable_cuboid` for more information.
    """

    func: Callable = deformable_shapes.spawn_deformable_cuboid

    size: tuple[float, float, float] = MISSING
    """Size of the cuboid."""


@configclass
class DeformableTrapezoidCfg(DeformableShapeCfg):
    """Configuration parameters for a cuboid prim.

    See :meth:`spawn_deformable_cuboid` for more information.
    """

    func: Callable = deformable_shapes.spawn_deformable_trapezoid

    size: tuple[float, float, float] = MISSING
    """Size of the cuboid."""

    top_width_ratio: float = 0.5
    """Ratio of the top width to the base width of the trapezoid."""

    base_width_ratio: float = 0.8
    """Ratio of the base width to the base width of the trapezoid."""
