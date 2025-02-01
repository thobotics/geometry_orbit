# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import omni.physics.tensors.impl.api as physx
from pxr import UsdPhysics

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
import omni.isaac.orbit.utils.string as string_utils

from .rigid_object import RigidObject
from .rigid_object_data import RigidObjectData

if TYPE_CHECKING:
    from .rigid_object_cfg import RigidObjectWithAttachmentCfg
    from omni.isaac.orbit.assets import JointAttachmentCfg


class RigidObjectWithAttachment(RigidObject):

    def __init__(self, cfg: RigidObjectWithAttachmentCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)

        # spawn body attachments
        if cfg.attachments is not None:
            self._spawn_joint_attachments(cfg.attachments)

    def _spawn_joint_attachments(self, attachments: Sequence[JointAttachmentCfg]):
        """Spawn attachments for the bodies."""
        for attachment in attachments:
            sim_utils.apply_joint_attachment(
                attachment.name,
                attachment.joint_type,
                attachment.prim_path,
                f"{self.cfg.prim_path}/mesh",
                local_pos0=attachment.local_pos0,
                local_pos1=attachment.local_pos1,
            )
