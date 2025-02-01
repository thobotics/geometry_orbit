# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pxr import UsdPhysics

import carb
import omni.physics.tensors.impl.api as physx
from omni.isaac.core.prims.rigid_prim_view import RigidPrimView
from omni.isaac.core.articulations.articulation_view import ArticulationView

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.math as math_utils
import omni.isaac.orbit.utils.string as string_utils

from ..asset_base import AssetBase
from .rope_data import RopeData

if TYPE_CHECKING:
    from ..body_attachment import JointAttachmentCfg
    from .rope_cfg import RopeCfg


class Rope(AssetBase):
    """A fixed object asset class."""

    cfg: RopeCfg
    """Configuration instance for the rigid object."""

    def __init__(self, cfg: RopeCfg):
        """Initialize the rigid object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        # container for data access
        self._data = RopeData()

        self._num_instances = 0
        self._num_links_list = []
        self._links_offset = []

        # spawn body attachments
        if cfg.attachments is not None:
            self._spawn_joint_attachments(cfg.attachments)

    """
    Properties
    """

    @property
    def data(self) -> RopeData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_links_total(self) -> int:
        """Number of bodies in the asset."""
        return sum(self.data._num_links_list)

    @property
    def link_names(self) -> list[str]:
        """Ordered names of bodies in articulation."""
        prim_paths = self.root_view.prim_paths
        return [path.split("/")[-1] for path in prim_paths]

    @property
    def root_view(self) -> physx.RigidBodyView:
        """The root view of the rigid object."""
        return self._root_view

    @property
    def root_physx_view(self) -> physx.RigidBodyView:
        """The root view of the rigid object."""
        return self._root_view

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        link_ids = self._env_ids_to_link_ids(env_ids)

        # reset link states
        self._data._link_state_w[link_ids] = self._data.default_link_state[link_ids][..., :7].clone()
        self._data._link_vel_w[link_ids] = torch.zeros_like(self._data._link_vel_w[link_ids])

        self.root_view.set_transforms(self._data._link_state_w, indices=link_ids)
        self.root_view.set_velocities(self._data._link_vel_w, indices=link_ids)

    def write_data_to_sim(self):
        pass

    def update(self):
        # -- root-state (note: we roll the quaternion to match the convention used in Isaac Sim -- wxyz)
        link_state = self.root_view.get_transforms()
        self._data._link_state_w[:, :7] = link_state
        # -- update common data
        self._update_common_data()

    def find_bodies(self, name_keys: str | Sequence[str]) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Please check the :meth:`omni.isaac.orbit.utils.string_utils.resolve_matching_names` function for more
        information on the name matching.

        Args:
            name_keys: A regular expression or a list of regular expressions to match the body names.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.link_names)

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
        raise RuntimeError(f"Writting root pose to simulation is not supported for {self.__class__.__name__}.")

    def write_link_pose_to_sim(self, link_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        link_ids = self._env_ids_to_link_ids(env_ids)
        self._data._link_state_w[link_ids] = link_state
        self.root_view.set_transforms(self._data._link_state_w, indices=link_ids)

    def write_link_velocity_to_sim(self, link_velocities: torch.Tensor, env_ids: Sequence[int] | None = None):
        link_ids = self._env_ids_to_link_ids(env_ids)
        self._data._link_vel_w[link_ids] = link_velocities
        self.root_view.set_velocities(self._data._link_vel_w, indices=link_ids)

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # create simulation view
        self._physics_sim_view = physx.create_simulation_view(self._backend)
        self._physics_sim_view.set_subspace_roots("/")

        # Get the environment information
        rope_prim_paths = sim_utils.find_matching_prim_paths(self.cfg.prim_path)
        if len(rope_prim_paths) == 0:
            raise RuntimeError(f"No prims found at path: {self.cfg.prim_path}.")

        self._num_instances = len(rope_prim_paths)

        # Extract the link prims
        root_prim_path_expr = self.cfg.prim_path + "/Link.*"
        self._root_view = self._physics_sim_view.create_rigid_body_view(root_prim_path_expr.replace(".*", "*"))

        # Initialize the links offset and number of links
        self.data._num_links_list = []
        self.data._links_offset = []
        max_links = 0
        offset = 0
        prim_paths = self._root_view.prim_paths
        for i in range(self._num_instances):
            prim_path_list = [path for path in prim_paths if path.startswith(rope_prim_paths[i])]
            self.data._num_links_list.append(len(prim_path_list))
            self.data._links_offset.append(offset)

            offset += len(prim_path_list)
            max_links = max(max_links, len(prim_path_list))

        self.data._max_links = max_links

        # log information about the articulation
        carb.log_info(f"Fixed body initialized at: {self.cfg.prim_path}.")
        carb.log_info(f"Number of instances: {self.num_instances}")
        carb.log_info(f"Number of links: {self.num_links_total}")
        carb.log_info(f"Link names: {self.link_names}")
        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)
        self._ALL_LINK_INDICES = torch.arange(self.root_view.count, dtype=torch.long, device=self.device)

        # asset data
        # -- properties
        self._data.link_names = self.link_names
        # -- root states
        self._data.root_state_w = torch.zeros(self.num_instances, 7, device=self.device)
        self._data.default_root_state = torch.zeros_like(self._data.root_state_w)
        # -- link states
        self._data._link_state_w = torch.zeros(self.num_links_total, 7, device=self.device)
        self._data.default_link_state = torch.zeros_like(self._data._link_state_w)
        # -- post-computed
        self._data._link_vel_w = torch.zeros(self.num_links_total, 6, device=self.device)

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
            self._data.default_link_state = torch.cat(
                [
                    self.root_view.get_transforms(),
                    self.root_view.get_velocities(),
                ],
                dim=-1,
            )

    def _update_common_data(self):
        """Update common quantities related to rigid objects.

        Note:
            This has been separated from the update function to allow for the child classes to
            override the update function without having to worry about updating the common data.
        """
        # -- link state
        link_state = self.root_view.get_transforms()
        self._data._link_state_w[..., :7] = link_state
        # -- link velocity
        link_vel = self.root_view.get_velocities()
        self._data._link_vel_w[..., :3] = link_vel[..., :3]
        self._data._link_vel_w[..., 3:6] = link_vel[..., 3:6]

    def _spawn_joint_attachments(self, attachments: Sequence[JointAttachmentCfg]):
        """Spawn attachments for the bodies."""
        for attachment in attachments:
            sim_utils.apply_joint_attachment(
                attachment.name,
                attachment.joint_type,
                attachment.prim_path,
                f"{self.cfg.prim_path}/Link{attachment.attached_link_idx}",
                local_pos0=attachment.local_pos0,
                local_pos1=attachment.local_pos1,
            )

    def _env_ids_to_link_ids(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            return self._ALL_LINK_INDICES

        """Convert environment indices to link indices."""
        link_ids = []
        for env_id in env_ids:
            link_ids.extend([env_id * self.data._max_links + i for i in range(self.data._num_links_list[env_id])])
        return torch.tensor(link_ids, dtype=torch.long, device=self.device)

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._root_view = None
