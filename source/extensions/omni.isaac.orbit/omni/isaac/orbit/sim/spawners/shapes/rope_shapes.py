# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import math
import numpy as np
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.objects.capsule import DynamicCapsule

import omni.isaac.core.utils.stage as stage_utils
from omni.physx.scripts import physicsUtils
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from omni.isaac.core.materials.physics_material import PhysicsMaterial
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.materials.preview_surface import PreviewSurface

from omni.isaac.orbit.sim import schemas
from omni.isaac.orbit.sim.utils import bind_physics_material, bind_visual_material, clone

if TYPE_CHECKING:
    from . import rope_shapes_cfg


@clone
def spawn_rope(
    prim_path: str,
    cfg: rope_shapes_cfg.RopeShapeCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
) -> Usd.Prim:
    """Spawn a deformable cuboid prim."""

    _spawn_rope_segments(prim_path, cfg, translation=translation, orientation=orientation)

    # return the prim
    return prim_utils.get_prim_at_path(prim_path)


"""
Helper functions.
"""


def _spawn_rope_segments(
    prim_path: str,
    cfg: rope_shapes_cfg.RopeShapeCfg,
    attributes: dict | None = None,
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

    # create the rope
    contact_offset = cfg.contact_offset
    rest_offset = cfg.rest_offset
    cone_angle_limit = cfg.cone_angle_limit
    rope_damping = cfg.rope_damping
    rope_stiffness = cfg.rope_stiffness
    inv_mass = cfg.inv_mass
    length = cfg.length
    num_links = cfg.num_links

    start = np.asarray(translation)
    direction = np.asarray([1.0, 0.0, 0.0])
    radius = cfg.radius
    color = np.array([0.02745, 0.156862, 0.20392])

    start = Gf.Vec3f(*start)
    direction = Gf.Vec3f(*direction)
    # make sure the direction is unit vector
    Gf.Normalize(direction)
    link_length = length / num_links
    link_mass = 1 / (inv_mass * num_links)

    axis = Gf.Cross(Gf.Vec3f(1, 0, 0), direction)
    w = Gf.GetLength(direction) + Gf.Dot(Gf.Vec3f(1, 0, 0), direction)
    link_orientation = np.asarray((w, axis[0], axis[1], axis[2]))
    # make sure the orientation is unit vector so that it represents a rotation
    norm = math.sqrt(Gf.GetLength(axis) ** 2 + w**2)
    link_orientation = link_orientation / norm

    physics_material_path = cfg.physics_material_path
    visual_material_path = cfg.visual_material_path

    physics_material = PhysicsMaterial(
        prim_path=physics_material_path,
        static_friction=0.2,
        dynamic_friction=1.0,
        restitution=0.0,
    )

    visual_material = PreviewSurface(prim_path=visual_material_path, color=color)

    links = []
    joints = []
    previous_path = None
    for linkInd in range(num_links):
        link_pos = np.asarray(start + direction * (link_length - radius) * linkInd)
        link_path = prim_path + f"/Link{linkInd}"
        link = _create_link(
            path=link_path,
            location=link_pos,
            orientation=link_orientation,
            link_half_length=link_length * 0.5,
            link_radius=radius,
            link_mass=link_mass,
            color=color,
            contact_offset=contact_offset,
            rest_offset=rest_offset,
            physics_material=physics_material,
            visual_material=visual_material,
        )
        if previous_path is not None:
            joint = _create_joint(
                previous_path=previous_path,
                link_path=link_path,
                link_half_length=link_length * 0.5,
                link_radius=radius,
                cone_angle_limit=cone_angle_limit,
                rope_damping=rope_damping,
                rope_stiffness=rope_stiffness,
            )
            joints.append(joint)
        links.append(link)
        previous_path = link_path

    return links, joints


def _create_link(
    path: str,
    location: np.array,
    orientation: np.array,
    link_half_length: float,
    link_radius: float,
    link_mass: float,
    color: np.array,
    contact_offset: float,
    rest_offset: float,
    physics_material: PhysicsMaterial,
    visual_material: VisualMaterial,
):
    """Create one link for rigid rope.
    Args:
        path (str): the path of the link, should be child of rope's path.
        location (np.ndarray): the translation of the link.
                                shape is (3, ).
        orientation (np.ndarray): quaternion of the link.
                                    quaternion is scalar-first (w, x, y, z). shape is (4, ).
        link_half_length (float): half of the length of one link.
        link_radius (float): the radius of the cross section of the link.
        link_mass (float): the mass of one link.
        color (np.ndarray): the color of the link.
                            shape is (3, ).
        contact_offset (float): the contact offset of the link.
        rest_offset (float): the rest offset of the link.
    """
    capsule = DynamicCapsule(
        path,
        translation=location,
        orientation=orientation,
        scale=np.asarray((1.0, 1.0, 1.0)),
        mass=link_mass,
        height=link_half_length,
        radius=link_radius,
        color=color,
        physics_material=physics_material,
        visual_material=visual_material,
    )
    capsuleGeom = UsdGeom.Capsule.Get(stage_utils.get_current_stage(), path)
    capsuleGeom.CreateAxisAttr("X")
    capsule.set_rest_offset(rest_offset)
    capsule.set_contact_offset(contact_offset)
    return capsule


def _create_joint(
    previous_path: str,
    link_path: str,
    link_half_length: float,
    link_radius: float,
    cone_angle_limit: float,
    rope_damping: float,
    rope_stiffness: float,
):
    """Create joint for two links at previous_path and link_path.
    Args:
        previous_path (str): the path of the first link.
        link_path (str): the path of the second link.
        link_half_length (float): half of the length of the link.
        link_radius (float): the radius of the cross section of the link.
        cone_angle_limit (int): The angle limits between two links in degree.
        rope_damping (float): The damping attribute of the rope.
        rope_stiffness (float): The stiffness attribute of the rope.
    """
    joint_path = link_path + "/SphericalJoint"
    joint = UsdPhysics.Joint.Define(stage_utils.get_current_stage(), joint_path)
    joint.CreateBody0Rel().SetTargets([previous_path])
    joint.CreateBody1Rel().SetTargets([link_path])

    joint_x = link_half_length - 0.5 * link_radius
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(joint_x, 0, 0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(-joint_x, 0, 0))

    # locked DOF (lock - low is greater than high)
    d6Prim = joint.GetPrim()
    limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transX")
    limitAPI.CreateLowAttr(1.0)
    limitAPI.CreateHighAttr(-1.0)
    limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transY")
    limitAPI.CreateLowAttr(1.0)
    limitAPI.CreateHighAttr(-1.0)
    limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "transZ")
    limitAPI.CreateLowAttr(1.0)
    limitAPI.CreateHighAttr(-1.0)
    limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, "rotX")
    limitAPI.CreateLowAttr(1.0)
    limitAPI.CreateHighAttr(-1.0)

    # Moving DOF:
    dofs = ["rotY", "rotZ"]
    for d in dofs:
        limitAPI = UsdPhysics.LimitAPI.Apply(d6Prim, d)
        limitAPI.CreateLowAttr(-cone_angle_limit)
        limitAPI.CreateHighAttr(cone_angle_limit)

        # joint drives for rope dynamics:
        driveAPI = UsdPhysics.DriveAPI.Apply(d6Prim, d)
        driveAPI.CreateTypeAttr("force")
        driveAPI.CreateDampingAttr(rope_damping)
        driveAPI.CreateStiffnessAttr(rope_stiffness)

    return joint
