# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import omni.physics.tensors.impl.api as physx
from omni.isaac.core.prims.soft.cloth_prim_view import ClothPrimView
from omni.isaac.core.prims.soft.particle_system_view import ParticleSystemView

import omni.isaac.orbit.sim as sim_utils
import omni.isaac.orbit.utils.string as string_utils

from ..asset_base import AssetBase
from .cloth_object_data import ClothObjectData

if TYPE_CHECKING:
    from ..body_attachment import BodyAttachmentCfg
    from .cloth_object_cfg import ClothObjectCfg


class ClothObject(AssetBase):
    """A cloth object asset class.

    Cloth objects are assets comprising of cloth objects. For now, only particle cloth is supported.
    To access the data returned by the PhysX simulation, use the :attr:`root_physx_view` attribute.
    In addition, we also provide a :attr:`root_particle_view` attribute to access the particle system view.

    .. note::
        In the current version, the `ClothPrimView`_ and `ParticleSystemView`_ class from Isaac Sim are
        employed to access the simulation data. The `ClothPrimView`_ class is used to access the simulation
        data of the cloth object, while the `ParticleSystemView`_ class is used to access the particle system
        data.
    .. _`ClothPrimView`: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html?highlight=cloth#cloth-prim-view
    .. _`ParticleSystemView`: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html?highlight=cloth#particle-system-view
    """

    cfg: ClothObjectCfg
    """Configuration instance for the cloth object."""

    def __init__(self, cfg: ClothObjectCfg):
        """Initialize the cloth object.

        Args:
            cfg: A configuration instance.
        """
        super().__init__(cfg)
        # container for data access
        self._data = ClothObjectData()

        # spawn body attachments
        if cfg.attachments is not None:
            self._spawn_body_attachments(cfg.attachments)

    """
    Properties
    """

    @property
    def device(self) -> str:
        """Memory device for computation."""
        return self.root_view._device

    @property
    def data(self) -> ClothObjectData:
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
    def root_view(self) -> ClothPrimView:
        """Cloth body view for the asset (Isaac Sim)."""
        return self._root_view

    @property
    def root_particle_view(self) -> ParticleSystemView:
        """Particle system view for the asset (Isaac Sim)."""
        return self._root_particle_view

    @property
    def num_points(self) -> int:
        """Number of points in the cloth mesh."""
        return self._num_points

    @property
    def root_state_w(self) -> torch.Tensor:
        """Root state `[pos, quat, lin_vel, ang_vel]` in simulation world frame. Shape: (num_prims, 13)."""
        return self._data.root_state_w

    @property
    def points_pos_w(self) -> torch.Tensor:
        """positions of the points used to simulate the mesh in world frame. Shape: (num_prims, num_points, 3)."""
        return self._data.points_pos_w

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
    def root_physx_view(self) -> physx.ParticleClothView:
        """Cloth body view for the asset (PhysX).

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
        self._data.points_state_w[:, :, :3] = self.root_view.get_world_positions()
        self._data.points_state_w[:, :, 3:] = self.root_view.get_velocities()

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

        The root state comprises of the particle positions and velocities. All the quantities are in the simulation frame.

        Args:
            root_state: Root state in simulation frame. Shape is ``(len(env_ids), max_particles_per_cloth, 3)``.
            env_ids: Environment indices. If :obj:`None`, then all indices are used.
        """
        # set into simulation
        self.write_root_pos_to_sim(root_state[..., :3], env_ids=env_ids)
        self.write_root_velocity_to_sim(root_state[..., 3:], env_ids=env_ids)

    def write_root_pos_to_sim(self, root_pos: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the root pos over selected environment indices into the simulation.

        The root pos comprises of the particle positions of the simulation mesh for the cloth.

        Args:
            root_pos: Root poses in simulation frame. Shape is ``(len(env_ids), max_particles_per_cloth, 3)``.
            env_ids: Environment indices. If :obj:`None`, then all indices are used.
        """
        # resolve all indices
        physx_env_ids = env_ids
        if env_ids is None:
            env_ids = slice(None)
            physx_env_ids = self._ALL_INDICES
        # note: we need to do this here since tensors are not set into simulation until step.
        # set into internal buffers
        self._data.points_state_w[env_ids, :, :3] = root_pos.clone()
        # set into simulation
        self.root_view.set_world_positions(self._data.points_state_w[env_ids, :, :3], indices=physx_env_ids)

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
        self._data.points_state_w[env_ids, :, 3:] = root_velocity.clone()
        # set into simulation
        self.root_view.set_velocities(self._data.points_state_w[:, :, 3:], indices=physx_env_ids)

    """
    Internal helper.
    """

    def _initialize_impl(self):

        if hasattr(self.cfg.spawn, "assets_cfg"):
            particle_path = self.cfg.spawn.assets_cfg[0].particle_system_props.particle_system_path
            cloth_path = self.cfg.spawn.assets_cfg[0].cloth_props.cloth_path
        else:
            particle_path = self.cfg.spawn.particle_system_props.particle_system_path
            cloth_path = self.cfg.spawn.cloth_props.cloth_path

        self._root_particle_view = ParticleSystemView(f"{self.cfg.prim_path}/{particle_path}")
        self._root_view = ClothPrimView(
            f"{self.cfg.prim_path}/{cloth_path}",
            reset_xform_properties=False,
        )
        self._root_particle_view.initialize()
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
        self._num_points = self.root_view.get_world_positions().size()[1]
        self._data.root_state_w = torch.zeros(self.root_view.count, 13, dtype=torch.float, device=self.device)
        self._data.points_state_w = torch.zeros(
            self.root_view.count,
            self.num_points,
            6,
            dtype=torch.float,
            device=self.device,
        )

        self._data.default_root_state_w = torch.zeros_like(self._data.root_state_w)
        self._data.default_points_state_w = torch.zeros_like(self._data.points_state_w)

    def _process_cfg(self):
        """Post processing of configuration parameters."""
        if type(self.cfg.init_state.pos) is list:
            id_map = self.cfg.spawn.geom_id_map(len(self._data.default_root_state_w), len(self.cfg.init_state.pos))
            for i in range(len(self._data.default_root_state_w)):
                self._data.default_root_state_w[i, :3] = torch.tensor(
                    self.cfg.init_state.pos[id_map[i]], device=self.device
                )
                self._data.default_root_state_w[i, 3:7] = torch.tensor(
                    self.cfg.init_state.rot[id_map[i]], device=self.device
                )
                self._data.default_root_state_w[i, 7:10] = torch.tensor(
                    self.cfg.init_state.lin_vel[id_map[i]], device=self.device
                )
                self._data.default_root_state_w[i, 10:] = torch.tensor(
                    self.cfg.init_state.ang_vel[id_map[i]], device=self.device
                )
        else:
            default_root_state = (
                self.cfg.init_state.pos
                + self.cfg.init_state.rot
                + self.cfg.init_state.lin_vel
                + self.cfg.init_state.ang_vel
            )
            self._data.default_root_state_w = torch.tensor(default_root_state, device=self.device).repeat(
                self.root_view.count, 1
            )

        self._data.default_points_state_w = torch.cat(
            [
                self.root_view.get_world_positions(indices=self._ALL_INDICES),
                self.root_view.get_velocities(indices=self._ALL_INDICES),
            ],
            dim=-1,
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

    """
    Internal simulation callbacks.
    """

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)
        # set all existing views to None to invalidate them
        self._physics_sim_view = None
        self._root_physx_view = None
        self._root_view = None
        self._root_particle_view = None
        self._body_physx_view = None
