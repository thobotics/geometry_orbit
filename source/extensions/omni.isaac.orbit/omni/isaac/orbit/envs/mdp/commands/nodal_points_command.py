# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import DeformableObject
from omni.isaac.orbit.managers import CommandTerm
from omni.isaac.orbit.markers import VisualizationMarkers, VisualizationMarkersCfg

if TYPE_CHECKING:
    from omni.isaac.orbit.envs import BaseEnv

    from .commands_cfg import UniformNodalPointsCommandCfg


class UniformNodalPointsCommand(CommandTerm):
    """Command generator for generating nodal position commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space.

    The position commands are generated in the local frame of the environment, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.


    """

    cfg: UniformNodalPointsCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformNodalPointsCommandCfg, env: BaseEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.deformable_mesh: DeformableObject = env.scene[cfg.asset_name]

        self.target_pos = torch.tensor(cfg.target_pos, device=self.device)

        if cfg.indices is None:
            self.set_nodal_indices, self.nodal_idx = cfg.select_indices_fnc(
                env, target_pos=self.target_pos, **cfg.select_indices_params
            )
        else:
            self.set_nodal_indices = None
            self.nodal_idx = torch.tensor(cfg.indices, device=self.device).tile(self.num_envs, 1)

        # create buffers
        # -- commands: (x, y, z) in root frame
        self.nodal_pos_command_b = torch.zeros(self.num_envs, self.nodal_idx.shape[1], 3, device=self.device)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pos command. Shape is (num_envs, 3 * len(self.nodal_idx))."""
        # transform the command to the local frame
        return self.nodal_pos_command_b.view(self.num_envs, -1)

    @property
    def nodal_indices(self) -> torch.Tensor:
        """The indices of the nodal points for which the commands are generated."""
        return self.nodal_idx

    """
    Implementation specific functions.
    """

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), self.nodal_idx.shape[1], device=self.device)
        self.nodal_pos_command_b[env_ids] = self.target_pos.unsqueeze(0)
        self.nodal_pos_command_b[env_ids, :, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.nodal_pos_command_b[env_ids, :, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        self.nodal_pos_command_b[env_ids, :, 2] += r.uniform_(*self.cfg.ranges.pos_z)

        # randomly select the cmd nodal indices
        if self.set_nodal_indices is not None:
            for id in env_ids:
                random_indices = torch.randint(0, len(self.set_nodal_indices[id]), ())
                self.nodal_idx[id] = self.set_nodal_indices[id][random_indices]

    def _update_command(self):
        pass

    def _update_metrics(self):
        # compute the error
        pos_error = self.deformable_mesh.data.nodal_pos_w[
            torch.arange(self.num_envs).view(-1, 1), self.nodal_idx, :3
        ] - (self.nodal_pos_command_b + self._env.scene.env_origins.unsqueeze(1))
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                # -- goal pose
                marker_cfg = VisualizationMarkersCfg(
                    prim_path="/Visuals/Command/goal_pose",
                    markers={
                        "sphere": sim_utils.SphereCfg(
                            radius=0.1,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                        ),
                    },
                )
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
                # -- current body pose
                marker_cfg = marker_cfg.copy()
                marker_cfg.prim_path = "/Visuals/Command/nodal_pos"
                marker_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 1.0)
                )
                self.nodal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            self.nodal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
                self.nodal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        # -- goal pose
        self.goal_pos_visualizer.visualize(
            (self.nodal_pos_command_b + self._env.scene.env_origins.unsqueeze(1)).view(-1, 3)
        )
        # -- current body pose
        nodal_pos_w = self.deformable_mesh.data.nodal_pos_w[torch.arange(self.num_envs).view(-1, 1), self.nodal_idx]
        self.nodal_pos_visualizer.visualize(nodal_pos_w.view(-1, 3))
