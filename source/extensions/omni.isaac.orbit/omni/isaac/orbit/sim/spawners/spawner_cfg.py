# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from pxr import Usd

import omni.isaac.orbit.sim.schemas as schemas
import omni.isaac.orbit.sim.spawners.materials as materials
from omni.isaac.orbit.utils import configclass


@configclass
class SpawnerCfg:
    """Configuration parameters for spawning an asset.

    Spawning an asset is done by calling the :attr:`func` function. The function takes in the
    prim path to spawn the asset at, the configuration instance and transformation, and returns the
    prim path of the spawned asset.

    The function is typically decorated with :func:`omni.isaac.orbit.sim.spawner.utils.clone` decorator
    that checks if input prim path is a regex expression and spawns the asset at all matching prims.
    For this, the decorator uses the Cloner API from Isaac Sim and handles the :attr:`copy_from_source`
    parameter.
    """

    func: Callable[..., Usd.Prim] = MISSING
    """Function to use for spawning the asset.

    The function takes in the prim path (or expression) to spawn the asset at, the configuration instance
    and transformation, and returns the source prim spawned.
    """

    visible: bool = True
    """Whether the spawned asset should be visible. Defaults to True."""

    semantic_tags: list[tuple[str, str]] | None = None
    """List of semantic tags to add to the spawned asset. Defaults to None,
    which means no semantic tags will be added.

    The semantic tags follow the `Replicator Semantic` tagging system. Each tag is a tuple of the
    form ``(type, data)``, where ``type`` is the type of the tag and ``data`` is the semantic label
    associated with the tag. For example, to annotate a spawned asset in the class avocado, the semantic
    tag would be ``[("class", "avocado")]``.

    You can specify multiple semantic tags by passing in a list of tags. For example, to annotate a
    spawned asset in the class avocado and the color green, the semantic tags would be
    ``[("class", "avocado"), ("color", "green")]``.

    .. seealso::

        For more information on the semantics filter, see the documentation for the `semantics schema editor`_.

    .. _semantics schema editor: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/semantics_schema_editor.html#semantics-filtering

    """

    copy_from_source: bool = True
    """Whether to copy the asset from the source prim or inherit it. Defaults to True.

    This parameter is only used when cloning prims. If False, then the asset will be inherited from
    the source prim, i.e. all USD changes to the source prim will be reflected in the cloned prims.

    .. versionadded:: 2023.1

        This parameter is only supported from Isaac Sim 2023.1 onwards. If you are using an older
        version of Isaac Sim, this parameter will be ignored.
    """


@configclass
class RigidObjectSpawnerCfg(SpawnerCfg):
    """Configuration parameters for spawning a rigid asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """

    mass_props: schemas.MassPropertiesCfg | None = None
    """Mass properties."""
    rigid_props: schemas.RigidBodyPropertiesCfg | None = None
    """Rigid body properties."""
    collision_props: schemas.CollisionPropertiesCfg | None = None
    """Properties to apply to all collision meshes."""

    activate_contact_sensors: bool = False
    """Activate contact reporting on all rigid bodies. Defaults to False.

    This adds the PhysxContactReporter API to all the rigid bodies in the given prim path and its children.
    """


@configclass
class RopeSpawnerCfg(SpawnerCfg):
    """Configuration parameters for spawning a deformable asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """

    mass_props: schemas.MassPropertiesCfg | None = None


@configclass
class DeformableObjectSpawnerCfg(SpawnerCfg):
    """Configuration parameters for spawning a deformable asset.

    Note:
        By default, all properties are set to None. This means that no properties will be added or modified
        to the prim outside of the properties available by default when spawning the prim.
    """

    mass_props: schemas.MassPropertiesCfg | None = None
    """Mass properties."""
    deformable_props: schemas.DeformableBodyPropertiesCfg | None = None
    """Deformable body properties."""
    attachment_props: schemas.AttachmentPropertiesCfg | None = None
    """Deformable attachment properties."""


class ClothObjectSpawnerCfg(SpawnerCfg):
    """Configuration parameters for spawning a cloth asset.
    See :meth:`omni.isaac.orbit.sim.spawners.cloth_object.spawn_cloth_object`
    for more information.
    """

    cloth_props: schemas.ClothPropertiesCfg = MISSING

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
