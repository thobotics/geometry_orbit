# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

# orbit
from omni.isaac.orbit.utils import configclass

from ..asset_base_cfg import AssetBaseCfg
from .cloth_object import ClothObject

if TYPE_CHECKING:
    from omni.isaac.orbit.assets import BodyAttachmentCfg


@configclass
class ClothObjectCfg(AssetBaseCfg):
    """Configuration parameters for a cloth object."""

    @configclass
    class InitialStateCfg:
        """Initial state of the Cloth body."""

        # root position
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x,y,z (m)
        rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # w,x,y,z
        lin_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x,y,z (m/s)
        ang_vel: tuple[float, float, float] = (0.0, 0.0, 0.0)  # x,y,z (rad/s)

    class_type: type = ClothObject

    init_state: InitialStateCfg = InitialStateCfg()
    """Initial state of the rigid object. Defaults to identity pose with zero velocity."""

    attachments: Sequence[BodyAttachmentCfg] | None = None
    """List of attachments to spawn with the deformable object."""
