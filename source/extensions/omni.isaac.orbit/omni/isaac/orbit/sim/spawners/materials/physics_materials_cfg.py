# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from omni.isaac.orbit.utils import configclass

from . import physics_materials


@configclass
class PhysicsMaterialCfg:
    """Configuration parameters for creating a physics material.

    Physics material are PhysX schemas that can be applied to a USD material prim to define the
    physical properties related to the material. For example, the friction coefficient, restitution
    coefficient, etc. For more information on physics material, please refer to the
    `PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/_build/physx/latest/class_px_base_material.html>`_.
    """

    func: Callable = MISSING
    """Function to use for creating the material."""


@configclass
class RigidBodyMaterialCfg(PhysicsMaterialCfg):
    """Physics material parameters for rigid bodies.

    See :meth:`spawn_rigid_body_material` for more information.

    Note:
        The default values are the `default values used by PhysX 5
        <https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#rigid-body-materials>`_.
    """

    func: Callable = physics_materials.spawn_rigid_body_material

    static_friction: float = 0.5
    """The static friction coefficient. Defaults to 0.5."""

    dynamic_friction: float = 0.5
    """The dynamic friction coefficient. Defaults to 0.5."""

    restitution: float = 0.0
    """The restitution coefficient. Defaults to 0.0."""

    improve_patch_friction: bool = True
    """Whether to enable patch friction. Defaults to True."""

    friction_combine_mode: Literal["average", "min", "multiply", "max"] = "average"
    """Determines the way friction will be combined during collisions. Defaults to `"average"`.

    .. attention::

        When two physics materials with different combine modes collide, the combine mode with the higher
        priority will be used. The priority order is provided `here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/_build/physx/latest/struct_px_combine_mode.html#pxcombinemode>`_.
    """

    restitution_combine_mode: Literal["average", "min", "multiply", "max"] = "average"
    """Determines the way restitution coefficient will be combined during collisions. Defaults to `"average"`.

    .. attention::

        When two physics materials with different combine modes collide, the combine mode with the higher
        priority will be used. The priority order is provided `here
        <https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/_build/physx/latest/struct_px_combine_mode.html#pxcombinemode>`_.
    """

    compliant_contact_stiffness: float = 0.0
    """Spring stiffness for a compliant contact model using implicit springs. Defaults to 0.0.

    A higher stiffness results in behavior closer to a rigid contact. The compliant contact model is only enabled
    if the stiffness is larger than 0.
    """

    compliant_contact_damping: float = 0.0
    """Damping coefficient for a compliant contact model using implicit springs. Defaults to 0.0.

    Irrelevant if compliant contacts are disabled when :obj:`compliant_contact_stiffness` is set to zero and
    rigid contacts are active.
    """


@configclass
class DeformableBodyMaterialCfg(PhysicsMaterialCfg):
    """Physics material parameters for deformable bodies.

    See :meth:`spawn_deformable_body_material` for more information.

    Note:
        The default values are the `default values used by PhysX 5
        <https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/deformable-bodies.html#deformable-body-material>`_.
    """

    func: Callable = physics_materials.spawn_deformable_body_material

    density: float | None = None
    """The material density. Autocomputed."""

    dynamic_friction: float = 0.25
    """The dynamic friction for the deformable material."""

    youngs_modulus: float = 50000000.0
    """The Youngs' modulus for the deformable material."""

    poissons_ratio: float = 0.45
    """The Poissons' ratio for the deformable material."""

    elasticity_damping: float = 0.005
    """The elasticity damping for the deformable material."""

    damping_scale: float = 1.0
    """The damping scale for the deformable material."""


@configclass
class ParticleMaterialCfg(PhysicsMaterialCfg):
    """Physics material parameters for particle systems.
    See :meth:`spawn_particle_system_material`
    for more information.
    """

    func: Callable = physics_materials.spawn_particle_system_material

    friction: float | None = None
    """The friction coefficient for the particle material."""

    particle_friction_scale: float | None = None
    """The particle friction scale for the particle material."""

    damping: float | None = None
    """The global damping velocity for the particle material."""

    drag: float | None = None
    """The drag coefficient for the particle material."""

    lift: float | None = None
    """The lift coefficient for the particle material."""

    adhesion: float | None = None
    """The adhesion for the particle material."""

    adhesion_offset_scale: float | None = None
    """The adhesion offset scale for the particle material."""

    particle_adhesion_scale: float | None = None
    """The particle adhesion scale for the particle material."""

    gravity_scale: float | None = None
    """The gravity scale for the particle material."""
