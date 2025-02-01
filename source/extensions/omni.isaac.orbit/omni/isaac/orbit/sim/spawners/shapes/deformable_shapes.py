# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import omni.isaac.core.utils.deformable_mesh_utils as deformableMeshUtils
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.physx.scripts import physicsUtils
from pxr import Gf, Usd, UsdGeom

from omni.isaac.orbit.sim import schemas
from omni.isaac.orbit.sim.utils import bind_physics_material, bind_visual_material, clone

from .mesh_utils import createTriangleMeshTrapezoid

if TYPE_CHECKING:
    from . import deformable_shapes_cfg


@clone
def spawn_deformable_cuboid(
    prim_path: str,
    cfg: deformable_shapes_cfg.DeformableCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn a deformable cuboid prim."""
    # resolve the scale
    size = min(cfg.size)
    scale = [dim for dim in cfg.size]
    # spawn cuboid if it doesn't exist.
    attributes = {"size": size}

    tri_points, tri_indices = deformableMeshUtils.createTriangleMeshCube(cfg.voxel_count)
    _spawn_deformable_geom_from_voxel(
        prim_path, cfg, tri_points, tri_indices, attributes, translation, orientation, scale
    )

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


@clone
def spawn_deformable_trapezoid(
    prim_path: str,
    cfg: deformable_shapes_cfg.DeformableTrapezoidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn a deformable cuboid prim."""
    # resolve the scale
    size = min(cfg.size)
    scale = [dim for dim in cfg.size]
    # spawn cuboid if it doesn't exist.
    attributes = {"size": size}

    tri_points, tri_indices = createTriangleMeshTrapezoid(cfg.voxel_count, cfg.top_width_ratio, cfg.base_width_ratio)
    _spawn_deformable_geom_from_voxel(
        prim_path, cfg, tri_points, tri_indices, attributes, translation, orientation, scale
    )

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


"""
Helper functions.
"""


def _spawn_deformable_geom_from_voxel(
    prim_path: str,
    cfg: deformable_shapes_cfg.GeometryCfg,
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
    mesh_prim_path = prim_path + "/mesh"
    mesh_prim = UsdGeom.Mesh.Define(stage_utils.get_current_stage(), mesh_prim_path)
    mesh_prim.GetPointsAttr().Set(tri_points)
    mesh_prim.GetFaceVertexIndicesAttr().Set(tri_indices)
    mesh_prim.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))

    # apply attributes
    physicsUtils.setup_transform_as_scale_orient_translate(mesh_prim)
    if translation is not None:
        physicsUtils.set_or_add_translate_op(mesh_prim, translation)
    if orientation is not None:
        physicsUtils.set_or_add_orient_op(mesh_prim, Gf.Quatf(*orientation))
    if scale is not None:
        physicsUtils.set_or_add_scale_op(mesh_prim, scale)

    # apply deformable properties
    if cfg.deformable_props is not None:
        schemas.modify_deformable_body_properties(prim_path, cfg.deformable_props)

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
    # apply physics material
    if cfg.physics_material is not None:
        if not cfg.physics_material_path.startswith("/"):
            material_path = f"{prim_path}/{cfg.physics_material_path}"
        else:
            material_path = cfg.physics_material_path
        # create material
        cfg.physics_material.func(material_path, cfg.physics_material)
        # apply material
        bind_physics_material(prim_path, material_path)

    # note: we apply rigid properties in the end to later make the instanceable prim
    # apply mass properties
    if cfg.mass_props is not None:
        schemas.define_mass_properties(prim_path, cfg.mass_props)
