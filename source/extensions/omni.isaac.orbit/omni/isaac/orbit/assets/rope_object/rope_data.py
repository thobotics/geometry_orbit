# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import dataclass


@dataclass
class RopeData:
    """Data container for a rigid object."""

    ##
    # Properties.
    ##

    link_names: list[str] = None
    """Body names in the order parsed by the simulation view."""

    ##
    # Default states.
    ##

    default_root_state: torch.Tensor = None
    """Default root state ``[pos, quat]`` in local environment frame. Shape is (count, 7)."""

    default_link_state: torch.Tensor = None
    """Default state of all links `[pos, quat, lin_vel, rot_vel]` in local environment frame.
    Shape is (count, num_bodies, 13)."""

    ##
    # Frame states.
    ##

    root_state_w: torch.Tensor = None
    """Root state ``[pos, quat]`` in simulation world frame. Shape is (count, 7)."""

    _link_state_w: torch.Tensor = None
    """State of all bodies `[pos, quat]` in simulation world frame.
    Shape is (count, num_bodies, 7)."""

    _link_vel_w: torch.Tensor = None
    """Velocities of all bodies. Shape is (count, num_bodies, 6).

    Note:
        This quantity is computed based on the rigid body state from the last step.
    """

    ##
    # Additional properties to handle variable-length link chains.
    ##

    _num_links_list: list[int] = None

    _links_offset: list[int] = None

    _max_links: int = None

    """
    Properties
    """

    @property
    def root_pos_w(self) -> torch.Tensor:
        """Root position in simulation world frame. Shape is (count, 3)."""
        return self.link_pos_w.mean(dim=1)

    @property
    def root_quat_w(self) -> torch.Tensor:
        """Root orientation (w, x, y, z) in simulation world frame. Shape is (count, 4)."""
        return self.root_state_w[:, 3:7]

    @property
    def default_root_pos(self) -> torch.Tensor:
        """Default root position in local environment frame. Shape is (count, 3)."""
        return self.default_link_pos.mean(dim=1)

    @property
    def default_link_pos(self) -> torch.Tensor:
        """Default positions of all bodies in local environment frame. Shape is (count, num_bodies, 3)."""
        link_pos = self.default_link_state[:, :3]
        return torch.stack(
            [
                link_pos[offset : offset + num_links]
                for offset, num_links in zip(self._links_offset, self._num_links_list)
            ],
            dim=0,
        )

    @property
    def default_link_rot(self) -> torch.Tensor:
        """Default orientation (w, x, y, z) of all bodies in local environment frame. Shape is (count, num_bodies, 4)."""
        link_rot = self.default_link_state[..., 3:7]
        return torch.stack(
            [
                link_rot[offset : offset + num_links]
                for offset, num_links in zip(self._links_offset, self._num_links_list)
            ],
            dim=0,
        )

    @property
    def link_pos_w(self) -> torch.Tensor:
        """Positions of all bodies in simulation world frame. Shape is (count, num_bodies, 3)."""
        link_pos = self._link_state_w[:, :3]
        return torch.stack(
            [
                link_pos[offset : offset + num_links]
                for offset, num_links in zip(self._links_offset, self._num_links_list)
            ],
            dim=0,
        )

    @property
    def link_quat_w(self) -> torch.Tensor:
        """Orientation (w, x, y, z) of all bodies in simulation world frame. Shape is (count, num_bodies, 4)."""
        link_quat = self._link_state_w[..., 3:7]
        return torch.stack(
            [
                link_quat[offset : offset + num_links]
                for offset, num_links in zip(self._links_offset, self._num_links_list)
            ],
            dim=0,
        )

    @property
    def link_vel_w(self) -> torch.Tensor:
        """Velocity of all bodies in simulation world frame. Shape is (count, num_bodies, 3)."""
        link_vel = self._link_vel_w[..., :3]
        return torch.stack(
            [
                link_vel[offset : offset + num_links]
                for offset, num_links in zip(self._links_offset, self._num_links_list)
            ],
            dim=0,
        )
