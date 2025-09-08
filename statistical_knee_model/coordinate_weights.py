"""Copyright (c) 2024, Stanford Neuromuscular Biomechanics Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
1. Redistributions of source code must retain the above copyright notice, 
this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the 
documentation and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from this
software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import vtk

from constants import (
    FEMUR_Z_POINT_THRESH,
    TIBIA_Z_POINT_THRESH
)

def fade_mesh_points(mesh_point_coords: Union[np.ndarray, torch.Tensor], 
                     fade_coord_thresh: float, fade_to: str, 
                     visualize=False) -> np.ndarray:
    """Fade the coordinates corresponding to the provided indices, fading to 
    zero in the direction specified.

    Args:
        mesh_point_coords: n_points x 3 array (where each column is an x, y, 
            or z coordinate).
        fade_coord_thresh: threshold at which to begin fading point 
            weights.
        fade_to: the end at which the weight should be zero: 'high' or 'low'.
        visualize: True to visualize the mesh and its weights. Defaults to 
            False.

    Returns:
        fading_weights: array of the weights associated with each of the 
            points.
    """
    if isinstance(mesh_point_coords, torch.Tensor):
        use_torch = True
    else:
        use_torch = False
    z = mesh_point_coords[:, 2]

    if fade_to == 'high':
        fade_coord_mask = z > fade_coord_thresh
        z_fade_coords = z[fade_coord_mask]

        min_fade_z = z_fade_coords.min() # assign a value of 1
        max_fade_z = z_fade_coords.max() # assign a value of 0

        min_fade_weight = 1
        max_fade_weight = 0

    elif fade_to == 'low':
        fade_coord_mask = z < fade_coord_thresh
        z_fade_coords = z[fade_coord_mask]
        
        min_fade_z = z_fade_coords.min() # assign a value of 0
        max_fade_z = z_fade_coords.max() # assign a value of 1

        min_fade_weight = 0
        max_fade_weight = 1
    else:
        raise ValueError('expected fade_to to be "high" or "low". Received '
                        f'{fade_to}.')
    
    # line coefficients
    m = (min_fade_weight - max_fade_weight) / (min_fade_z - max_fade_z)
    b = min_fade_weight - m * min_fade_z

    if use_torch:
        coord_weights = torch.ones(mesh_point_coords.shape[0], 
                                   dtype=torch.float32)
        coord_weights = torch.where(fade_coord_mask, m * z + b, 
                                    coord_weights)
        coord_weights = torch.where(coord_weights < 0, 0, coord_weights)
    else:
        coord_weights = np.ones(mesh_point_coords.shape[0])
        coord_weights[fade_coord_mask] = m * z_fade_coords + b
        coord_weights[coord_weights < 0] = 0

    if visualize:
        y = mesh_point_coords[:, 1]
        plt.figure()
        scatter = plt.scatter(y, z, alpha=0.5, s=1, c=coord_weights)
        plt.colorbar(scatter)
        plt.show()
    return coord_weights


def concat_weights(femur_weights: np.ndarray,
                   tibia_weights: np.ndarray,
                   patella_weights: np.ndarray
                   ) -> np.ndarray:
    """Format the coordinate weights for downstream use.
    
    Args:
        femur_weights: array of femur weights.
        tibia_weights: array of tibia weights.
        patella_weights: array of patella weights.

    Returns:
        concat_weights: array of concatenated weights.
    """
    if isinstance(femur_weights, torch.Tensor):
        use_torch = True
    else:
        use_torch = False

    if use_torch:
        concat_weights = torch.cat((femur_weights, tibia_weights, 
                   patella_weights))
    else:
        concat_weights = np.concatenate((femur_weights, 
                                        tibia_weights, 
                                        patella_weights))
    return concat_weights


def compute_point_weights(points: Union[np.ndarray, torch.Tensor], mean_meshes: List[vtk.vtkPolyData]
                          ) -> np.ndarray:
    """Compute the weights associated with the mesh points.
    
    Args:
        points: points (n_points x 3), in order of [femur, tibia, patella]
        mean_meshes: list of mean meshes for each bone.

    Returns:
        coord_weights: array of length n_joint_coords containing point 
            weights.
    """
    n_mesh_points = []
    for mesh in mean_meshes:
        n_mesh_points.append(mesh.GetNumberOfPoints())
    points_tibia_start_idx = n_mesh_points[0]
    points_patella_start_idx = n_mesh_points[0] + n_mesh_points[1]

    femur_points = points[:points_tibia_start_idx]
    tibia_points = points[points_tibia_start_idx:points_patella_start_idx]

    femur_point_weights = fade_mesh_points(femur_points, 
                    fade_coord_thresh=FEMUR_Z_POINT_THRESH, fade_to='high')
    tibia_point_weights = fade_mesh_points(tibia_points, 
                    fade_coord_thresh=TIBIA_Z_POINT_THRESH, fade_to='low')
    if isinstance(femur_point_weights, torch.Tensor):
        patella_point_weights = torch.ones(n_mesh_points[2], 
                                           dtype=torch.float32)
    else:
        patella_point_weights = np.ones(n_mesh_points[2])
    coord_weights = concat_weights(femur_point_weights, 
                                   tibia_point_weights,
                                   patella_point_weights
                                   )
    return coord_weights
