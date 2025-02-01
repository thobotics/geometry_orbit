# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for spawning primitive shapes in the simulation.

NVIDIA Omniverse provides various primitive shapes that can be used to create USDGeom prims. Based
on the configuration, the spawned prim can be:

* a visual mesh (no physics)
* a static collider (no rigid body)
* a rigid body (with collision and rigid body properties).

"""

from .cloth_shapes import spawn_cloth_with_holes, spawn_plain_cloth
from .cloth_shapes_cfg import ParticleClothCfg, SquareClothWithHoles
from .deformable_shapes import spawn_deformable_cuboid, spawn_deformable_trapezoid
from .deformable_shapes_cfg import DeformableCuboidCfg, DeformableShapeCfg, DeformableTrapezoidCfg
from .rope_shapes import spawn_rope
from .rope_shapes_cfg import RopeShapeCfg
from .shapes import spawn_capsule, spawn_cone, spawn_cuboid, spawn_cylinder, spawn_sphere
from .shapes_cfg import CapsuleCfg, ConeCfg, CuboidCfg, CylinderCfg, ShapeCfg, SphereCfg
