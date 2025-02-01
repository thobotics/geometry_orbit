# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from omni.isaac.orbit.sim.schemas import AttachmentPropertiesCfg
from omni.isaac.orbit.utils import configclass


@configclass
class BodyAttachmentCfg:
    """Configuration parameters for spawning an attachment."""

    prim_path: str | None = None
    """The prim path to spawn the attachment at."""
    name: str | None = None
    """The name of the attachment prim."""
    attachment_props: AttachmentPropertiesCfg | None = None
    """Attachment properties."""


@configclass
class JointAttachmentCfg:
    """Configuration parameters for spawning a joint attachment."""

    name: str = "joint_attachment"
    """The prim path to spawn the attachment at."""
    joint_type: str | None = None
    """The joint type."""
    prim_path: str | None = None
    """The parent prim path."""
    attached_link_idx: int | None = None
    """The index of the link to attach to."""
    local_pos0: tuple[float, float, float] | None = None
    """Local position of the first attachment point in the root frame. Defaults to (0.0, 0.0, 0.0)."""
    local_pos1: tuple[float, float, float] | None = None
    """Local position of the second attachment point in the root frame. Defaults to (0.0, 0.0, 0.0)."""

