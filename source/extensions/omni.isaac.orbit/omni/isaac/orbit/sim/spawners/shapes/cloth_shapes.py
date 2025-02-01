# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.physx.scripts import deformableUtils, physicsUtils
from pxr import Usd, UsdGeom

from omni.isaac.orbit.sim import schemas
from omni.isaac.orbit.sim.utils import bind_physics_material, bind_visual_material, clone

from .mesh_utils import create_triangle_mesh_square_with_holes

if TYPE_CHECKING:
    from . import cloth_shapes_cfg


@clone
def spawn_plain_cloth(
    prim_path: str,
    cfg: cloth_shapes_cfg.ParticleClothCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn a deformable cuboid prim."""
    size = cfg.size if cfg.size is not None else (5, 5)
    attributes = {"size": size}

    tri_points, tri_indices = deformableUtils.create_triangle_mesh_square(dimx=size[0], dimy=size[1], scale=1.0)
    _spawn_particle_cloth_from_triangles(prim_path, cfg, tri_points, tri_indices, attributes, translation, orientation)

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_cloth_with_holes(
    prim_path: str,
    cfg: cloth_shapes_cfg.SquareClothWithHoles,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn a deformable cuboid prim."""
    size = cfg.size if cfg.size is not None else (5, 5)
    attributes = {"size": size}

    tri_points, tri_indices = create_triangle_mesh_square_with_holes(
        dimx=size[0], dimy=size[1], holes=cfg.holes, scale=1.0
    )
    _spawn_particle_cloth_from_triangles(prim_path, cfg, tri_points, tri_indices, attributes, translation, orientation)

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


"""
Helper functions.
"""


def _spawn_particle_cloth_from_triangles(
    prim_path: str,
    cfg: cloth_shapes_cfg.ParticleClothCfg,
    tri_points: list[float],
    tri_indices: list[int],
    attributes: dict,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
):
    """Create a USDGeom-based prim with the given attributes."""
    # spawn geometry if it doesn't exist.
    if not prim_utils.is_prim_path_valid(prim_path):
        prim_utils.create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    else:
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    # create the mesh prim
    mesh_prim_path = f"{prim_path}/{cfg.cloth_props.cloth_path}"
    mesh_prim = UsdGeom.Mesh.Define(stage_utils.get_current_stage(), mesh_prim_path)
    mesh_prim.GetPointsAttr().Set(tri_points)
    mesh_prim.GetFaceVertexIndicesAttr().Set(tri_indices)
    mesh_prim.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))

    # apply attributes
    physicsUtils.setup_transform_as_scale_orient_translate(mesh_prim)
    if scale is not None:
        physicsUtils.set_or_add_scale_op(mesh_prim, scale)

    if cfg.particle_system_props is not None:
        particle_system_path, physx_particle_system_api = schemas.define_particle_system_properties(
            prim_path, cfg.particle_system_props
        )

    # apply physics material
    if cfg.particle_material is not None:
        if not cfg.particle_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.particle_material_path}"
        else:
            material_path = cfg.particle_material_path
        # create material
        cfg.particle_material.func(material_path, cfg.particle_material)
        # apply material
        bind_physics_material(particle_system_path, material_path)

    # apply particle properties
    if cfg.cloth_props is not None and cfg.particle_system_props is not None:
        schemas.modify_particle_cloth_properties(prim_path, particle_system_path, cfg.cloth_props)

    # apply visual material
    if cfg.visual_material is not None:
        if not cfg.visual_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.visual_material_path}"
        else:
            material_path = cfg.visual_material_path
        # create material
        cfg.visual_material.func(material_path, cfg.visual_material)
        # apply material
        bind_visual_material(prim_path, material_path)

    # apply mass properties
    if cfg.mass_props is not None:
        schemas.define_mass_properties(mesh_prim_path, cfg.mass_props)
