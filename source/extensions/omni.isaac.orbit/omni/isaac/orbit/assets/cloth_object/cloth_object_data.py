# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class ClothObjectData:
    """Data container for a robot."""

    body_names: list[str] | None = None

    ##
    # Frame states.
    ##

    root_state_w: torch.Tensor | None = None
    """Root state `[pos, quat, lin_evel, rot_vel]` in simulation world frame. Shape: (num_prims, 13)."""

    points_state_w: torch.Tensor | None = None
    """Point state used to simulate the mesh in world frame. Shape: (num_prims, num_points, 6)."""

    ##
    # Default states.
    ##

    default_root_state_w: torch.Tensor | None = None
    """Default state ``[pos, quat, lin_evel, rot_vel]`` in simulation world frame.
    Shape is ``(num_prims,  13)``.
    """

    default_points_state_w: torch.Tensor | None = None
    """Default point state used to simulate the mesh in world frame. Shape: (num_prims, num_points, 6)."""

    ##
    # Frame states.
    ##

    particle_masses: torch.Tensor | None = None
    """Masses of the particles. Shape: (num_prims, num_points)."""

    pressures: torch.Tensor | None = None
    """Pressures of the particles. Shape: (num_prims, num_points).
    Note: if > 0, a particle cloth has an additional pressure constraint that provides
    inflatable (i.e. balloon-like) dynamics. The pressure times the rest volume defines
    the volume the inflatable tries to match. Pressure only works well for closed or
    approximately closed meshes, range: [0, inf), units: dimensionless
    """

    particle_groups: torch.Tensor | None = None
    """Group Id of the particles of each prim, range: [0, 2^20). Shape: (num_prims, num_points)."""

    self_collisions: torch.Tensor | None = None
    """Enable Self-collision of the particles. Shape: (num_prims, num_points)."""

    self_collision_filters: torch.Tensor | None = None
    """Whether the simulation should filter particle-particle collisions based on the
    rest position distances . Shape: (num_prims, num_points)."""

    stretch_stiffnesses: torch.Tensor | None = None
    """represents the stretch spring stiffnesses for linear springs placed between particles
    to counteract stretching, shape is (N,). range: [0, inf),
    units: force / distance = mass / second / second. Shape: (num_prims, num_points)."""

    bend_stiffnesses: torch.Tensor | None = None
    """Bend stiffnesses of the particles. Shape: (num_prims, num_points).
    Represents the spring bend stiffnesses for linear springs placed in a way to
    counteract bending,  shape is (N,). range: [0, inf),
    units: force / distance = mass / second / second
    """

    shear_stiffnesses: torch.Tensor | None = None
    """Shear stiffnesses of the particles. Shape: (num_prims, num_points).
    Represents the shear stiffnesses for linear springs placed in a way to
    counteract shear,  shape is (N,). range: [0, inf),
    units: force / distance = mass / second / second
    """

    spring_dampings: torch.Tensor | None = None
    """Spring dampings of the particles. Shape: (num_prims, num_points).
    Damping on cloth spring constraints. Applies to all constraints
    parameterized by stiffness attributes, range: [0, inf),  shape is (N,).
    units: force * second / distance = mass / second
    """

    """
    Properties
    """

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape: (num_prims, 3)."""
        return self.root_state_w[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation in simulation world frame. Shape: (num_prims, 4)."""
        return self.root_state_w[:, 3:7]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        """Root linear velocity in simulation world frame. Shape: (num_prims, 3)."""
        return self.root_state_w[:, 7:10]

    @property
    def root_rot_vel_w(self) -> torch.Tensor:
        """Root rotational velocity in simulation world frame. Shape: (num_prims, 3)."""
        return self.root_state_w[:, 10:13]

    @property
    def points_pos_w(self) -> torch.Tensor:
        """Positions of the points used to simulate the mesh in world frame. Shape: (num_prims, num_points, 3)."""
        return self.points_state_w[:, :, :3]

    @property
    def points_vel_w(self) -> torch.Tensor:
        """Velocities of the points used to simulate the mesh in world frame. Shape: (num_prims, num_points, 3)."""
        return self.points_state_w[:, :, 3:6]
