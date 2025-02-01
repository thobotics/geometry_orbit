# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class FixedObjectData:
    """Data container for a rigid object."""

    ##
    # Properties.
    ##

    body_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    ##
    # Default states.
    ##

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat]`` in local environment frame. Shape is (count, 7)."""

    ##
    # Frame states.
    ##

    root_state_w: torch.Tensor = None
    """Root state ``[pos, quat]`` in simulation world frame. Shape is (count, 7)."""

    heading_w: torch.Tensor = None
    """Yaw heading of the base frame (in radians). Shape is (count,).

    Note:
        This quantity is computed by assuming that the forward-direction of the base
        frame is along x-direction, i.e. :math:`(1, 0, 0)`.
    """

    body_state_w: torch.Tensor = None
    """State of all bodies `[pos, quat]` in simulation world frame.
    Shape is (count, num_bodies, 7)."""

    body_vel_w: torch.Tensor = None
    """Velocities of all bodies. Shape is (count, num_bodies, 6).

    Note:
        This quantity is computed based on the rigid body state from the last step.
    """

    """
    Properties
    """

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (count, 3)."""
        return self.root_state_w[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (count, 4)."""
        return self.root_state_w[:, 3:7]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (count, num_bodies, 3)."""
        return self.body_state_w[..., :3]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (count, num_bodies, 4)."""
        return self.body_state_w[..., 3:7]
