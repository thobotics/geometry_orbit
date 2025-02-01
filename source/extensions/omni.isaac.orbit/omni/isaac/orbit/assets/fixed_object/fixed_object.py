# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
from omni.isaac.core.prims.geometry_prim_view import GeometryPrimView

import omni.isaac.orbit.utils.math as math_utils
import omni.isaac.orbit.utils.string as string_utils

from ..asset_base import AssetBase
from .fixed_object_data import FixedObjectData

if TYPE_CHECKING:
    from .fixed_object_cfg import FixedObjectCfg


class FixedObject(AssetBase):
    """A fixed object asset class."""

    cfg: FixedObjectCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: FixedObjectCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        # container for data access
        self._data = FixedObjectData()

    """
    Properties
    """

    @property
    def data(self) -> FixedObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_view.count

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset."""
        return 1

    @property
    def body_names(self) -> list[str]:
        """Ordered names of bodies in articulation."""
        prim_paths = self.root_view.prim_paths[: self.num_bodies]
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def root_view(self) -> GeometryPrimView:
        """The root view of the rigid object."""
        return self._root_view

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # resolve all indices
        if env_ids is None:
            env_ids = slice(None)
        # reset last body vel
        self._last_body_state_w[env_ids] = 0.0

    def write_data_to_sim(self):
        pass

    def update(self, dt: float):
        # -- root-state (note: we roll the quaternion to match the convention used in Isaac Sim -- wxyz)
        root_state = self.root_view.get_world_poses()
        root_state = torch.cat(root_state, dim=-1)
        self._data.root_state_w[:, :7] = root_state
        # -- update common data
        self._update_common_data(dt)

    def find_bodies(self, name_keys: str | Sequence[str]) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`omni.isaac.orbit.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names)

    """
    Operations - Write to simulation.
    """

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the cartesian position, quaternion orientation in (w, x, y, z), and linear
        and angular velocity. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is (len(env_ids), 13).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # set into simulation
        self.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pose over selected environment indices into the simulation.

        The root pose comprises of the cartesian position and quaternion orientation in (w, x, y, z).

        Args:
            root_pose: Root poses in simulation frame. Shape is (len(env_ids), 7).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.root_state_w[env_ids, :7] = root_pose.clone()
        # convert root quaternion from wxyz to xyzw
        root_poses_wxyz = self._data.root_state_w[:, :7].clone()
        # set into simulation
        self.root_view.set_world_poses(
            positions=root_poses_wxyz[:, :3], orientations=root_poses_wxyz[:, 3:], indices=physx_env_ids
        )

    """
    Internal helper.
    """

    def _initialize_impl(self):
        self._root_view = GeometryPrimView(self.cfg.prim_path, reset_xform_properties=False)
        self._root_view.initialize()

        # log information about the articulation
        carb.log_info(f"Fixed body initialized at: {self.cfg.prim_path}.")
        carb.log_info(f"Number of instances: {self.num_instances}")
        carb.log_info(f"Number of bodies: {self.num_bodies}")
        carb.log_info(f"Body names: {self.body_names}")
        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        self._ALL_BODY_INDICES = torch.arange(self.root_view.count, dtype=torch.long, device=self.device)
        self.FORWARD_VEC_B = torch.tensor((1.0, 0.0, 0.0), device=self.device).repeat(self.num_instances, 1)

        # asset data
        # -- properties
        self._data.body_names = self.body_names
        # -- root states
        self._data.root_state_w = torch.zeros(self.num_instances, 7, device=self.device)
        self._data.default_root_state = torch.zeros_like(self._data.root_state_w)
        # -- body states
        self._data.body_state_w = torch.zeros(self.num_instances, self.num_bodies, 7, device=self.device)
        # -- post-computed
        self._data.heading_w = torch.zeros(self.num_instances, device=self.device)
        self._data.body_vel_w = torch.zeros(self.num_instances, self.num_bodies, 6, device=self.device)

        # history buffers for quantities
        # -- used to compute body accelerations numerically
        self._last_body_state_w = torch.zeros(self.num_instances, self.num_bodies, 7, device=self.device)

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        # default state
        # -- root state
        # note: we cast to tuple to avoid torch/numpy type mismatch.
        if type(self.cfg.init_state.pos) is list:
            id_map = self.cfg.spawn.geom_id_map(len(self._data.default_root_state), len(self.cfg.init_state.pos))
            for i in range(len(self._data.default_root_state)):
                self._data.default_root_state[i, :3] = torch.tensor(
                    self.cfg.init_state.pos[id_map[i]], device=self.device
                )
                self._data.default_root_state[i, 3:] = torch.tensor(
                    self.cfg.init_state.rot[id_map[i]], device=self.device
                )
        else:
            default_root_state = tuple(self.cfg.init_state.pos) + tuple(self.cfg.init_state.rot)
            default_root_state = torch.tensor(default_root_state, dtype=torch.float, device=self.device)
            self._data.default_root_state = default_root_state.repeat(self.num_instances, 1)

    def _update_common_data(self, dt: float):
        """Update common quantities related to rigid objects.

        Note:
            This has been separated from the update function to allow for the child classes to
            override the update function without having to worry about updating the common data.
        """
        # -- body-state (note: we roll the quaternion to match the convention used in Isaac Sim -- wxyz)
        # TODO: Now root state is the same as body state. Should we keep it like this?
        root_state = self.root_view.get_world_poses()
        root_state = torch.cat(root_state, dim=-1)
        self._data.body_state_w[..., :7] = root_state.view(-1, self.num_bodies, 7)
        # -- body velocity
        self._data.body_vel_w[..., :3] = (self._data.body_state_w[..., :3] - self._last_body_state_w[..., :3]) / dt
        self._data.body_vel_w[..., 3:6] = (
            torch.stack(math_utils.euler_xyz_from_quat(self._data.body_state_w[..., 3:7].view(-1, 4)), dim=-1)
            - torch.stack(math_utils.euler_xyz_from_quat(self._last_body_state_w[..., 3:7].view(-1, 4)), dim=-1)
        ).view(-1, self.num_bodies, 3) / dt
        self._last_body_state_w[:] = self._data.body_state_w
        # -- heading direction of root
        forward_w = math_utils.quat_apply(self._data.root_quat_w, self.FORWARD_VEC_B)
        self._data.heading_w[:] = torch.atan2(forward_w[:, 1], forward_w[:, 0])

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._root_view = None
