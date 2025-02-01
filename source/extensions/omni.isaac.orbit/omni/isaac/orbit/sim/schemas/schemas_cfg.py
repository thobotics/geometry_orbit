# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Literal

from collections.abc import Sequence

from omni.isaac.orbit.utils import configclass


@configclass
class ArticulationRootPropertiesCfg:
    """Properties to apply to the root of an articulation.

    See :meth:`modify_articulation_root_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    articulation_enabled: bool | None = None
    """Whether to enable or disable articulation."""
    enabled_self_collisions: bool | None = None
    """Whether to enable or disable self-collisions."""
    solver_position_iteration_count: int | None = None
    """Solver position iteration counts for the body."""
    solver_velocity_iteration_count: int | None = None
    """Solver position iteration counts for the body."""
    sleep_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an actor may go to sleep."""
    stabilization_threshold: float | None = None
    """The mass-normalized kinetic energy threshold below which an articulation may participate in stabilization."""


@configclass
class RigidBodyPropertiesCfg:
    """Properties to apply to a rigid body.

    See :meth:`modify_rigid_body_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    rigid_body_enabled: bool | None = None
    """Whether to enable or disable the rigid body."""
    kinematic_enabled: bool | None = None
    """Determines whether the body is kinematic or not.

    A kinematic body is a body that is moved through animated poses or through user defined poses. The simulation
    still derives velocities for the kinematic body based on the external motion.

    For more information on kinematic bodies, please refer to the `documentation <https://openusd.org/release/wp_rigid_body_physics.html#kinematic-bodies>`_.
    """
    disable_gravity: bool | None = None
    """Disable gravity for the actor."""
    linear_damping: float | None = None
    """Linear damping for the body."""
    angular_damping: float | None = None
    """Angular damping for the body."""
    max_linear_velocity: float | None = None
    """Maximum linear velocity for rigid bodies (in m/s)."""
    max_angular_velocity: float | None = None
    """Maximum angular velocity for rigid bodies (in rad/s)."""
    max_depenetration_velocity: float | None = None
    """Maximum depenetration velocity permitted to be introduced by the solver (in m/s)."""
    max_contact_impulse: float | None = None
    """The limit on the impulse that may be applied at a contact."""
    enable_gyroscopic_forces: bool | None = None
    """Enables computation of gyroscopic forces on the rigid body."""
    retain_accelerations: bool | None = None
    """Carries over forces/accelerations over sub-steps."""
    solver_position_iteration_count: int | None = None
    """Solver position iteration counts for the body."""
    solver_velocity_iteration_count: int | None = None
    """Solver position iteration counts for the body."""
    sleep_threshold: float | None = None
    """Mass-normalized kinetic energy threshold below which an actor may go to sleep."""
    stabilization_threshold: float | None = None
    """The mass-normalized kinetic energy threshold below which an actor may participate in stabilization."""
    locked_pos_axis: int | None = None
    """Locks translation along the specified axis. It is a binary encoding of the axis to lock."""
    locked_rot_axis: int | None = None
    """Locks rotation about the specified axis. It is a binary encoding of the axis to lock."""


@configclass
class ClothPropertiesCfg:
    """Properties to apply to the cloth prim.

    For more information on cloth properties,
    please refer to the `documentation <https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html?highlight=cloth#cloth-prim>`_.
    """

    particle_mass: float | None = 0.01
    """the mass of one single particle."""
    pressure: float | None = None
    """if > 0, a particle cloth has an additional pressure constraint that provides
    inflatable (i.e. balloon-like) dynamics. The pressure times the rest volume
    defines the volume the inflatable tries to match. Pressure only works well for
    closed or approximately closed meshes, range: [0, inf), units: dimensionless"""
    particle_group: int | None = 0
    """group Id of the particles, range: [0, 2^20)"""
    self_collision: bool | None = None
    """enable self collision of the particles or of the particle object."""
    self_collision_filter: bool | None = None
    """whether the simulation should filter particle-particle collisions
    based on the rest position distances."""
    spring_stretch_stiffness: float | None = None
    """represents a stiffness for linear springs placed between particles to
    counteract stretching, range: [0, inf), units: force / distance = mass / second / second"""
    spring_bend_stiffness: float | None = None
    """represents a stiffness for linear springs placed in a way to counteract
    bending, range: [0, inf), units: force / distance = mass / second / second"""
    spring_shear_stiffness: float | None = None
    """represents a stiffness for linear springs placed in a way to counteract
    shear, range: [0, inf), units: force / distance = mass / second / second"""
    spring_damping: float | None = None
    """damping on cloth spring constraints. Applies to all constraints
    parameterized by stiffness attributes, range: [0, inf),
    units: force * second / distance = mass / second"""
    dynamic_mesh_path: str | None = None
    """Relative path to the cloth mesh."""
    cloth_path: str | None = "mesh"
    """Relative path to spawn the cloth."""


@configclass
class ParticleSystemPropertiesCfg:
    """Properties to apply to the particle system."""

    particle_system_enabled: bool | None = None
    """whether to enable the particle system or not."""
    simulation_owner: str | None = None
    """single PhysicsScene that simulates this particle system."""
    contact_offset: float | None = None
    """Contact offset used for collisions with non-particle
    objects such as rigid or deformable bodies."""
    rest_offset: float | None = None
    """Rest offset used for collisions with non-particle objects
    such as rigid or deformable bodies."""
    particle_contact_offset: float | None = None
    """Contact offset used for interactions
    between particles. Must be larger than solid and fluid rest offsets."""
    solid_rest_offset: float | None = None
    """Rest offset used for solid-solid or solid-fluid
    particle interactions. Must be smaller than particle contact offset."""
    fluid_rest_offset: float | None = None
    """Rest offset used for fluid-fluid particle interactions.
    Must be smaller than particle contact offset."""
    enable_ccd: bool | None = None
    """Enable continuous collision detection for particles to help
    avoid tunneling effects."""
    solver_position_iteration_count: float | None = None
    """Number of solver iterations for position."""
    max_depenetration_velocity: float | None = None
    """The maximum velocity permitted to be introduced
    by the solver to depenetrate intersecting particles."""
    wind: Sequence[float] | None = None
    """wind applied to the particle system."""
    max_neighborhood: int | None = None
    """The particle neighborhood size."""
    max_velocity: float | None = None
    """Maximum particle velocity."""
    global_self_collision_enabled: bool | None = None
    """If True, self collisions follow
    particle-object-specific settings. If False, all particle self collisions are disabled, regardless
    of any other settings. Improves performance if self collisions are not needed."""
    non_particle_collision_enabled: float | None = None
    """Enable or disable particle collision with
    non-particle objects for all particles in the system. Improves performance if non-particle collisions
    are not needed."""
    particle_system_path: str | None = "particle_system"
    """Relative path to spawn the particle system."""


@configclass
class CollisionPropertiesCfg:
    """Properties to apply to colliders in a rigid body.

    See :meth:`modify_collision_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    collision_enabled: bool | None = None
    """Whether to enable or disable collisions."""
    contact_offset: float | None = None
    """Contact offset for the collision shape (in m).

    The collision detector generates contact points as soon as two shapes get closer than the sum of their
    contact offsets. This quantity should be non-negative which means that contact generation can potentially start
    before the shapes actually penetrate.
    """
    rest_offset: float | None = None
    """Rest offset for the collision shape (in m).

    The rest offset quantifies how close a shape gets to others at rest, At rest, the distance between two
    vertically stacked objects is the sum of their rest offsets. If a pair of shapes have a positive rest
    offset, the shapes will be separated at rest by an air gap.
    """
    torsional_patch_radius: float | None = None
    """Radius of the contact patch for applying torsional friction (in m).

    It is used to approximate rotational friction introduced by the compression of contacting surfaces.
    If the radius is zero, no torsional friction is applied.
    """
    min_torsional_patch_radius: float | None = None
    """Minimum radius of the contact patch for applying torsional friction (in m)."""


@configclass
class MassPropertiesCfg:
    """Properties to define explicit mass properties of a rigid body.

    See :meth:`modify_mass_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    mass: float | None = None
    """The mass of the rigid body (in kg).

    Note:
        If non-zero, the mass is ignored and the density is used to compute the mass.
    """
    density: float | None = None
    """The density of the rigid body (in kg/m^3).

    The density indirectly defines the mass of the rigid body. It is generally computed using the collision
    approximation of the body.
    """


@configclass
class JointDrivePropertiesCfg:
    """Properties to define the drive mechanism of a joint.

    See :meth:`modify_joint_drive_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    drive_type: Literal["force", "acceleration"] | None = None
    """Joint drive type to apply.

    If the drive type is "force", then the joint is driven by a force. If the drive type is "acceleration",
    then the joint is driven by an acceleration (usually used for kinematic joints).
    """


@configclass
class FixedTendonPropertiesCfg:
    """Properties to define fixed tendons of an articulation.

    See :meth:`modify_fixed_tendon_properties` for more information.

    .. note::
        If the values are None, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    tendon_enabled: bool | None = None
    """Whether to enable or disable the tendon."""
    stiffness: float | None = None
    """Spring stiffness term acting on the tendon's length."""
    damping: float | None = None
    """The damping term acting on both the tendon length and the tendon-length limits."""
    limit_stiffness: float | None = None
    """Limit stiffness term acting on the tendon's length limits."""
    offset: float | None = None
    """Length offset term for the tendon.

    It defines an amount to be added to the accumulated length computed for the tendon. This allows the application
    to actuate the tendon by shortening or lengthening it.
    """
    rest_length: float | None = None
    """Spring rest length of the tendon."""


@configclass
class DeformableBodyPropertiesCfg:
    """Properties to apply to a deformable body.

    See :meth:`set_deformable_body_properties` for more information.

    .. note::
        If the values are :obj:`None`, they are not modified. This is useful when you want to set only a subset of
        the properties and leave the rest as-is.
    """

    deformable_enabled: bool | None = None
    """Whether to enable or disable the deformable body."""
    kinematic_enabled: bool | None = None
    """Determines whether the body is kinematic or not."""
    collision_simplification: bool | None = None
    """Collision simplification enabled or not for the deformable body."""
    self_collision: bool | None = None
    """Self collision enabled or not for the deformable body."""
    simulation_hexahedral_resolution: int | None = None
    """The parameter controlling the resolution of the soft body simulation mesh."""
    vertex_velocity_damping: float | None = None
    """Velocity damping parameter controlling how much after every time step the nodal velocity is reduced."""
    solver_position_iteration_count: int | None = None
    """Number of the solver's positional iteration counts. Range: [1,255]"""
    sleep_threshold: float | None = None
    """Threshold that defines the maximal magnitude of the linear motion a soft body can move in one second
    such that it can go to sleep in the next frame. Range: [1,inf)
    """
    settling_threshold: float | None = None
    """Threshold that defines the maximal magnitude of the linear motion a soft body can move in one second
    such that it can go to sleep in the next frame. Range: [1,inf)
    """
    sleep_damping: float | None = None
    """Damping value that damps the motion of bodies that move slow enough to be candidates for sleeping. Range: [1,inf)"""
    self_collision_filter_distance: float | None = None
    """Penetration value that needs to get exceeded before contacts for self collision are generated.
    Will only have an effect if self collisions are enabled based on the rest position distances.
    """


@configclass
class AttachmentPropertiesCfg:
    """Properties to apply to an attachment."""

    attachment_enabled: bool | None = None
    """Whether to enable or disable the attachment."""
    actor0: str | None = None
    """The first actor to attach to."""
    actor1: str | None = None
    """The second actor to attach to."""
    points0: Sequence[float] | None = None
    """Attachment points in Actor 0 local space, defined in the actor's rest state, if it is deformable. Elements correspond one-to-one to elements in points1 attribute."""
    points1: Sequence[float] | None = None
    """Attachment points in Actor 1 local space, defined in the actor's rest state, if it is deformable. Elements correspond one-to-one to elements in points0 attribute."""
    collision_filter_indices0: Sequence[int] | None = None
    """Indices to geometry of Actor 0 that should not generate collisions with Actor 1 as specified by filterType0. Ignored for rigid bodies."""
    filter_type0: int | None = None
    """Specify if indices in collisionFilterIndices0 correspond to vertices; or mesh cell-geometry, i.e. triangles, tetrahedrons, etc."""
    collision_filter_indices1: Sequence[int] | None = None
    """Indices to mesh triangle/tet/hex/etc. of Actor 1 that should not generate collisions with Actor 0. Ignored for rigid bodies."""
    filter_type1: int | None = None
    """Specify if indices in collisionFilterIndices1 correspond to vertices; or mesh cell-geometry, i.e. triangles, tetrahedrons, etc."""
