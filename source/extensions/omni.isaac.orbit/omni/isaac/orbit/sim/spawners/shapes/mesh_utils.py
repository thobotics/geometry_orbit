# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy

from omni.physx.scripts import deformableUtils
from pxr import Gf


def cubeTetrahedra():
    tetra = []
    tetra.append([(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1)])
    tetra.append([(0, 0, 0), (1, 0, 1), (1, 1, 0), (0, 1, 1)])
    tetra.append([(0, 0, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)])
    tetra.append([(1, 0, 1), (1, 1, 1), (1, 1, 0), (0, 1, 1)])
    tetra.append([(0, 0, 0), (1, 1, 0), (0, 1, 0), (0, 1, 1)])
    return tetra


def createTetraVoxels(voxel_dim, occupancy_filter_func, occupancy_filter_args=None):
    dimx, dimy, dimz = voxel_dim, voxel_dim, voxel_dim

    grid = numpy.zeros((dimx, dimy, dimz), dtype="bool")

    # write voxel grid cell occupancy
    num_voxels = 0
    for (x, y, z), _ in numpy.ndenumerate(grid):
        if occupancy_filter_func(x, y, z, dimx, dimy, dimz, *occupancy_filter_args):
            grid[x][y][z] = True
            num_voxels = num_voxels + 1

    # create vertex grid to compact list map
    grid_to_indices = numpy.full((dimx + 1, dimy + 1, dimz + 1), -1, dtype="int32")

    index = 0
    for (x, y, z), _ in numpy.ndenumerate(grid_to_indices):
        # check adjacent cells
        (x_b, x_e) = (max(x - 1, 0), min(x + 1, dimx))
        (y_b, y_e) = (max(y - 1, 0), min(y + 1, dimy))
        (z_b, z_e) = (max(z - 1, 0), min(z + 1, dimz))
        neighbors = grid[x_b:x_e, y_b:y_e, z_b:z_e]
        if numpy.any(neighbors):
            grid_to_indices[x][y][z] = index
            index = index + 1

    # write points
    points = [0] * index
    for (x, y, z), index in numpy.ndenumerate(grid_to_indices):
        if index > -1:
            points[index] = Gf.Vec3f(x, y, z)

    # write tetra indices
    cube_tetra = cubeTetrahedra()
    indices = [0] * num_voxels * len(cube_tetra) * 4
    index = 0
    for (x, y, z), occupied in numpy.ndenumerate(grid):
        if occupied:
            mx, my, mz = x % 2, y % 2, z % 2
            flip = (mx + my + mz) % 2
            for src_tet in cube_tetra:
                tet = [-1] * 4
                if flip:
                    # flip tetrahedron if cube got mirrored an odd times
                    tet[0], tet[1], tet[2], tet[3] = src_tet[1], src_tet[0], src_tet[2], src_tet[3]
                else:
                    tet = src_tet

                for cx, cy, cz in tet:
                    # mirror every other cube across all dimensions
                    wx = mx + (1 - 2 * mx) * cx
                    wy = my + (1 - 2 * my) * cy
                    wz = mz + (1 - 2 * mz) * cz
                    indices[index] = int(grid_to_indices[x + wx][y + wy][z + wz])
                    index = index + 1

    return points, indices


def voxel_trapezoid_test(x, y, z, dimx, dimy, dimz, top_width_ratio, base_width_ratio):
    # Base width of the trapezium at the bottom
    base_width = min(dimx, dimy) * base_width_ratio
    # Top width of the trapezium (smaller than the base width)
    top_width = min(dimx, dimy) * top_width_ratio
    # Linear interpolation to calculate the current width at height z
    current_width = base_width - ((base_width - top_width) * (z / dimz))

    # Calculate the offset from the edges of the grid at the current z level
    offset = (min(dimx, dimy) - current_width) / 2

    return (offset < x < dimx - offset) and (offset < y < dimy - offset)


def createTetraVoxelTrapezoid(voxel_dim, top_width_ratio, base_width_ratio):
    args = (top_width_ratio, base_width_ratio)
    points, indices = createTetraVoxels(voxel_dim, voxel_trapezoid_test, args)
    voxel_dim_inv = 1.0 / voxel_dim
    for i in range(len(points)):
        points[i] = (points[i] * voxel_dim_inv) - Gf.Vec3f(0.5, 0.5, 0.5)
    return points, indices


def createTriangleMeshTrapezoid(dim: int, top_width_ratio: float, base_width_ratio: float):
    points, indices = createTetraVoxelTrapezoid(dim, top_width_ratio, base_width_ratio)
    tri_points, tri_indices = deformableUtils.extractTriangleSurfaceFromTetra(points, indices)
    return tri_points, tri_indices


# For surface mesh generation


def create_triangle_mesh_square_with_holes(dimx, dimy, holes, scale=1.0):
    """
    Creates points and vertex data for a regular-grid flat triangle mesh square with multiple holes, using Gf.Vec3f.

    Args:
        dimx: Mesh-vertex resolution in X
        dimy: Mesh-vertex resolution in Y
        holes: List of tuples defining holes in the format (center_x, center_y, radius)
        scale: Uniform scale applied to vertices

    Returns:
        points, indices: The vertex and index data
    """

    points = []
    indices = []

    # Check if a point is inside any of the holes
    def is_in_hole(x, y):
        return any(numpy.sqrt((x - cx) ** 2 + (y - cy) ** 2) < r for cx, cy, r in holes)

    # Generate points, skipping those inside holes
    for y in range(dimy + 1):
        for x in range(dimx + 1):
            if not is_in_hole(x, y):
                points.append(Gf.Vec3f(x, y, 0.0))

    # Map from old vertex indices to new ones, skipping those removed for holes
    index_map = {}
    for new_index, point in enumerate(points):
        index_map[point[1] * (dimx + 1) + point[0]] = new_index

    # Generate indices, adapting for holes
    for y in range(dimy):
        for x in range(dimx):
            if is_in_hole(x, y) or is_in_hole(x + 1, y) or is_in_hole(x, y + 1) or is_in_hole(x + 1, y + 1):
                continue

            v0 = index_map.get(y * (dimx + 1) + x)
            v1 = index_map.get(y * (dimx + 1) + x + 1)
            v2 = index_map.get((y + 1) * (dimx + 1) + x)
            v3 = index_map.get((y + 1) * (dimx + 1) + x + 1)

            if v0 is None or v1 is None or v2 is None or v3 is None:
                continue

            if (x % 2 == 0) != (y % 2 == 0):
                indices.extend([v0, v1, v2, v1, v3, v2])
            else:
                indices.extend([v0, v1, v3, v0, v3, v2])

    # Scale and center the mesh
    for i, point in enumerate(points):
        points[i] = Gf.Vec3f((point[0] / dimx - 0.5) * scale, (point[1] / dimy - 0.5) * scale, point[2] * scale)

    return points, indices
