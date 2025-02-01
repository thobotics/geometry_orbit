# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from omni.isaac.orbit.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .deformable_object import DeformableObject

if TYPE_CHECKING:
    from omni.isaac.orbit.assets import BodyAttachmentCfg


@configclass
class DeformableObjectCfg(AssetBaseCfg):
    """Configuration parameters for a deformable object."""

    class_type: type = DeformableObject

    attachments: Sequence[BodyAttachmentCfg] | None = None
    """List of attachments to spawn with the deformable object."""
