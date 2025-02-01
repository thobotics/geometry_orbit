# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import omni.physics.tensors.impl.api as physx
from omni.isaac.core.prims.soft.deformable_prim_view import DeformablePrimView

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.string as string_utils

from ..asset_base import AssetBase
from .deformable_object_data import DeformableObjectData

if TYPE_CHECKING:
    from ..body_attachment import BodyAttachmentCfg
    from .deformable_object_cfg import DeformableObjectCfg


class DeformableObject(AssetBase):
    """Class for handling deformable objects."""

    cfg: DeformableObjectCfg
    """Configuration instance for the deformable object."""

    def __init__(self, cfg: DeformableObjectCfg):
        """Initialize the deformable object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        # container for data access
        self._data = DeformableObjectData()

        # spawn body attachments
        if cfg.attachments is not None:
            self._spawn_body_attachments(cfg.attachments)

    """
    Properties
    """

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self.root_view._device  # pyright: ignore [reportPrivateUsage]

    @property
    def data(self) -> DeformableObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self.root_physx_view.count

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
    def root_view(self) -> DeformablePrimView:
        """Deformable body view for the asset (Isaac Sim)."""
        return self._root_view

    @property
    def root_physx_view(self) -> physx.SoftBodyView:
        """Deformable body view for the asset (PhysX).

        Note:
            Use this view with caution. It requires handling of tensors in a specific way.
        """
        return self._root_physx_view

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # resolve all indices
        if env_ids is None:
            env_ids = ...

    def write_data_to_sim(self):
        pass

    def update(self):
        self._data.nodal_state_w[:, : self.root_view.max_simulation_mesh_vertices_per_body, :] = (
            self.root_view.get_simulation_mesh_nodal_positions()
        )
        self._data.nodal_state_w[:, self.root_view.max_simulation_mesh_vertices_per_body :, :] = (
            self.root_view.get_simulation_mesh_nodal_velocities()
        )

        self._data.sim_element_rotations = self.root_view.get_simulation_mesh_element_rotations()
        self._data.collision_element_rotations = self.root_view.get_collision_mesh_element_rotations()
        self._data.sim_element_deformation_gradients = (
            self.root_view.get_simulation_mesh_element_deformation_gradients()
        )
        self._data.collision_element_deformation_gradients = (
            self.root_view.get_collision_mesh_element_deformation_gradients()
        )
        self._data.sim_element_stresses = self.root_view.get_simulation_mesh_element_stresses()
        self._data.collision_element_stresses = self.root_view.get_collision_mesh_element_stresses()

    def find_bodies(self, name_keys: str | Sequence[str]) -> tuple[list[int], list[str]]:
        """Find bodies in the articulation based on the name keys.

        Args:
            name_keys: A regular expression or a list of regular expressions
                to match the body names.

        Returns:
            A tuple of lists containing the body indices and names.
        """
        return string_utils.resolve_matching_names(name_keys, self.body_names)

    """
    Operations - Write to simulation.
    """

    def write_root_state_to_sim(self, root_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root state over selected environment indices into the simulation.

        The root state comprises of the nodal positions and velocities. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is ``(len(env_ids), 2*max_simulation_mesh_vertices_per_body, 3)``.
            env_ids: Environment indices. If :obj:`None`, then all indices are used.
        """
        # set into simulation
        self.write_root_pos_to_sim(
            root_state[:, : self.root_view.max_simulation_mesh_vertices_per_body, :], env_ids=env_ids
        )
        self.write_root_velocity_to_sim(
            root_state[:, self.root_view.max_simulation_mesh_vertices_per_body :, :], env_ids=env_ids
        )

    def write_root_pos_to_sim(self, root_pos: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pos over selected environment indices into the simulation.

        The root pos comprises of the nodal positions of the simulation mesh for the deformable body.

        Args:
            root_pos: Root poses in simulation frame. Shape is ``(len(env_ids), max_simulation_mesh_vertices_per_body, 3)``.
            env_ids: Environment indices. If :obj:`None`, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.nodal_state_w[env_ids, : self.root_view.max_simulation_mesh_vertices_per_body, :] = root_pos.clone()
        # set into simulation
        self.root_view.set_simulation_mesh_nodal_positions(
            self._data.nodal_state_w[env_ids, : self.root_view.max_simulation_mesh_vertices_per_body, :],
            indices=physx_env_ids,
        )

    def write_root_velocity_to_sim(self, root_velocity: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root velocity over selected environment indices into the simulation.

        The root velocity comprises of the nodal velocities of the simulation mesh for the deformable body.

        Args:
            root_velocity: Root velocities in simulation frame. Shape is ``(len(env_ids), max_simulation_mesh_vertices_per_body, 3)``.
            env_ids: Environment indices. If :obj:`None`, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.nodal_state_w[env_ids, self.root_view.max_simulation_mesh_vertices_per_body :, :] = (
            root_velocity.clone()
        )
        # set into simulation
        self.root_view.set_simulation_mesh_nodal_velocities(
            self._data.nodal_state_w[env_ids, self.root_view.max_simulation_mesh_vertices_per_body :, :],
            indices=physx_env_ids,
        )

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # -- object views
        self._root_view = DeformablePrimView(self.cfg.prim_path + "/.*", reset_xform_properties=False)
        self._root_view.initialize()
        # physx view is already initialized inside the DefomablePrimView
        self._physics_sim_view = self._root_view._physics_sim_view  # pyright: ignore [reportPrivateUsage]
        self._root_physx_view = self._root_view._physics_view  # pyright: ignore [reportPrivateUsage]
        # log information about the articulation
        carb.log_info(f"Deformable body initialized at: {self.cfg.prim_path}")
        carb.log_info(f"Number of bodies (orbit): {self.num_bodies}")
        carb.log_info(f"Body names (orbit): {self.body_names}")
        # create buffers
        self._create_buffers()
        # process configuration
        self._process_cfg()

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.root_view.count, dtype=torch.long, device=self.device)
        # asset data
        # -- properties
        self._data.body_names = self.body_names
        # -- root states
        self._data.nodal_state_w = torch.zeros(
            self.root_view.count,
            2 * self.root_view.max_simulation_mesh_vertices_per_body,
            3,
            dtype=torch.float,
            device=self.device,
        )
        self._data.default_nodal_state_w = torch.zeros_like(self._data.nodal_state_w)
        # -- element-wise data
        self._data.sim_element_rotations = torch.zeros(
            self.root_view.count,
            self.root_view.max_simulation_mesh_elements_per_body,
            4,
            dtype=torch.float,
            device=self.device,
        )
        self._data.collision_element_rotations = torch.zeros(
            self.root_view.count,
            self.root_view.max_collision_mesh_elements_per_body,
            4,
            dtype=torch.float,
            device=self.device,
        )
        self._data.sim_element_deformation_gradients = torch.zeros(
            self.root_view.count,
            self.root_view.max_simulation_mesh_elements_per_body,
            3,
            3,
            dtype=torch.float,
            device=self.device,
        )
        self._data.collision_element_deformation_gradients = torch.zeros(
            self.root_view.count,
            self.root_view.max_collision_mesh_elements_per_body,
            3,
            3,
            dtype=torch.float,
            device=self.device,
        )
        self._data.sim_element_stresses = torch.zeros(
            self.root_view.count,
            self.root_view.max_simulation_mesh_elements_per_body,
            3,
            3,
            dtype=torch.float,
            device=self.device,
        )
        self._data.collision_element_stresses = torch.zeros(
            self.root_view.count,
            self.root_view.max_collision_mesh_elements_per_body,
            3,
            3,
            dtype=torch.float,
            device=self.device,
        )

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        self._data.default_nodal_state_w[:, : self.root_view.max_simulation_mesh_vertices_per_body, :] = (
            self.root_view.get_simulation_mesh_nodal_positions()
        )

    def _spawn_body_attachments(self, attachments: Sequence[BodyAttachmentCfg]):
        """Spawn attachments for the bodies."""
        for attachment in attachments:
            sim_utils.apply_body_attachment(
                self.cfg.prim_path,
                attachment.prim_path,
                attachment.name,
                attachment.attachment_props,
            )
