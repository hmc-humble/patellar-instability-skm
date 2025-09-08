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
import os
from typing import Union

import numpy as np
import pandas as pd
from pymskt.mesh import Mesh
from pymskt.statistics.pca import create_vtk_mesh_from_deformed_points
from pytorch3d.loss.chamfer import (
    _handle_pointcloud_input,
    _validate_chamfer_reduction_inputs
)
from pytorch3d.ops.knn import (
    knn_gather,
    knn_points
)
import torch

from constants import (
    MESH_PARAMS,
    FEMUR_Z_POINT_THRESH,
    TIBIA_Z_POINT_THRESH
)
from coordinate_weights import (
    compute_point_weights,
    concat_weights,
    fade_mesh_points
)
from mesh import get_meshes
from registration_transforms import (
    get_transform_components,
    invert_transform
)
from StatisticalKneeModel import StatisticalKneeModel


SKM_DIR = os.path.join(os.path.dirname(__file__), 'model_results')

N_PCS = 34 # from Horn's parallel analysis

N_SAMPLES_RECON = 2000
MIN_N_EPOCHS = 1000
MAX_N_EPOCHS = 2000
N_MESH_SAMPLES_TO_FIT = 2000
LOSS_THRESH = 12 # chamfer distance threshold for a good fit


def _chamfer_distance_single_direction(
    x,
    y,
    x_lengths,
    y_lengths,
    x_normals,
    y_normals,
    x_point_weights,
    batch_weights,
    batch_reduction: Union[str, None],
    point_reduction: Union[str, None],
    norm: int,
    abs_cosine: bool,
):
    """Single-direction chamfer distance between two pointclouds x and y. 
    Based on the pytorch3d implementation, this method additionally allows for
    coordinate/point weighting.
    """
    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if x_point_weights is not None:
        if x_point_weights.size(0) != P1:
            raise ValueError("point weights must be of shape (P1,).")
        if not (x_point_weights >= 0).all():
            raise ValueError("point weights cannot be negative.")
        if x_point_weights.sum() == 0.0:
            x_point_weights = x_point_weights.view(1, P1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((0, 2)) * x_point_weights).sum() * 0.0, 
                    (x.sum((0, 2)) * x_point_weights).sum() * 0.0,
                )
            return ((x.sum((0, 2)) * x_point_weights) * 0.0, 
                    (x.sum((0, 2)) * x_point_weights) * 0.0)

    if batch_weights is not None:
        if batch_weights.size(0) != N:
            raise ValueError("batch weights must be of shape (N,).")
        if not (batch_weights >= 0).all():
            raise ValueError("batch weights cannot be negative.")
        if batch_weights.sum() == 0.0:
            batch_weights = batch_weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * batch_weights).sum() * 0.0,
                    (x.sum((1, 2)) * batch_weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * batch_weights) * 0.0, 
                    (x.sum((1, 2)) * batch_weights) * 0.0)

    cham_norm_x = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, 
                      K=1)
    cham_x = x_nn.dists[..., 0]  # (N, P1)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0

    if x_point_weights is not None:
        cham_x *= x_point_weights.view(1, P1)

    if batch_weights is not None:
        cham_x *= batch_weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]

        cosine_sim = torch.nn.functional.cosine_similarity(x_normals, 
            x_normals_near, dim=2, eps=1e-6)
        # If abs_cosine, ignore orientation and take the absolute value of the 
        # cosine sim.
        cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine 
                           else cosine_sim)

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0

        if x_point_weights is not None:
            cham_norm_x *= x_point_weights.view(1, P1)

        if batch_weights is not None:
            cham_norm_x *= batch_weights.view(N, 1)

    if point_reduction is not None:
        # Apply point reduction
        cham_x = cham_x.sum(1)  # (N,)
        if return_normals:
            cham_norm_x = cham_norm_x.sum(1)  # (N,)
        if point_reduction == "mean":
            x_lengths_clamped = x_lengths.clamp(min=1)
            cham_x /= x_lengths_clamped
            if return_normals:
                cham_norm_x /= div

        if batch_reduction is not None:
            # batch_reduction == "sum"
            cham_x = cham_x.sum()
            if return_normals:
                cham_norm_x = cham_norm_x.sum()
            if batch_reduction == "mean":
                div = batch_weights.sum() if batch_weights is not None else max(N, 1)
                cham_x /= div
                if return_normals:
                    cham_norm_x /= div

    cham_dist = cham_x
    cham_normals = cham_norm_x if return_normals else None
    return cham_dist, cham_normals

def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    x_point_weights=None,
    y_point_weights=None,
    batch_weights=None,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: Union[str, None] = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
):
    """Chamfer distance between two pointclouds x and y. Based on the 
    pytorch3d implementation, this method additionally allows for coordinate/
    point weighting.

    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object 
            representing a batch of point clouds with at most P1 points in 
            each batch element, batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object 
            representing a batch of point clouds with at most P2 points in 
            each batch element, batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of 
            points in each cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of 
            points in each cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        x_point_weights: Optional FloatTensor of shape (P1,) giving weights 
            for the P1 points in x.
        y_point_weights: Optional FloatTensor of shape (P2,) giving weights 
            for the P2 points in y.
        batch_weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"] or None.
        norm: int indicates the norm used for the distance. Supports 1 for L1 
            and 2 for L2.
        single_directional: If False (default), loss comes from both the 
            distance between each point in x and its nearest neighbor in y and
            each point in y and its nearest neighbor in x. If True, loss is 
            the distance between each point in x and its nearest neighbor in 
            y.
        abs_cosine: If False, loss_normals is from one minus the cosine 
            similarity. If True (default), loss_normals is from one minus the 
            absolute value of the cosine similarity, which means that exactly 
            opposite normals are considered equivalent to exactly matching 
            normals, i.e. sign does not matter.

    Returns:
        2-element tuple containing

        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y. If point_reduction is None, a 
          2-element tuple of Tensors containing forward and backward loss 
          terms shaped (N, P1) and (N, P2) (if single_directional is False) or
          a Tensor containing loss terms shaped (N, P1) (if single_directional 
          is True) is returned.
        - **loss_normals**: Tensor giving the reduced cosine distance of 
          normals between pointclouds in x and pointclouds in y. Returns None 
          if x_normals and y_normals are None. If point_reduction is None, a 
          2-element tuple of Tensors containing forward and backward loss 
          terms shaped (N, P1) and (N, P2) (if single_directional is False) or
          a Tensor containing loss terms shaped (N, P1) (if single_directional
          is True) is returned.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    if not ((norm == 1) or (norm == 2)):
        raise ValueError("Support for 1 or 2 norm.")
    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    cham_x, cham_norm_x = _chamfer_distance_single_direction(
        x,
        y,
        x_lengths,
        y_lengths,
        x_normals,
        y_normals,
        x_point_weights,
        batch_weights,
        batch_reduction,
        point_reduction,
        norm,
        abs_cosine,
    )
    if single_directional:
        return cham_x, cham_norm_x
    else:
        cham_y, cham_norm_y = _chamfer_distance_single_direction(
            y,
            x,
            y_lengths,
            x_lengths,
            y_normals,
            x_normals,
            y_point_weights,
            batch_weights,
            batch_reduction,
            point_reduction,
            norm,
            abs_cosine,
        )
        if point_reduction is not None:
            return (
                cham_x + cham_y, (cham_norm_x + cham_norm_y) 
                if cham_norm_x is not None else None,
            )
        return (
            (cham_x, cham_y), (cham_norm_x, cham_norm_y) 
            if cham_norm_x is not None else None,
        )


class KneeReconstruction():
    """Class to reconstruct the joint by optimizing the joint alignment 
    parameters and shape features that result in the best fit."""
    def __init__(self, seg_filename: str, right_knee: bool, verbose: bool = False):
        """
        Args:
            seg_filename: path to the segmentation file.
            right_knee: True if the segmentation is of a right knee, False 
                otherwise.
            verbose: True to print information. Defaults to False.
        """
        self.seg_filename = seg_filename
        self.right_knee = right_knee

        self.seg_dir = os.path.dirname(seg_filename)
        self.sample_dir = os.path.split(self.seg_dir)[0]
        self.recon_save_dir = os.path.join(self.sample_dir, 'recon')
        self.fitted_params_filename = os.path.join(self.recon_save_dir, 
                                              'fitted_params.csv')

        self.fitted_params = None
        self.verbose = verbose
        self.log = pd.DataFrame()

    def get_mesh_transforms_and_coords(self):
        """Run rigid registration. Along the way, extract gross (to align 
        femur) and individual bone transforms. Following rigid registration, 
        extract point coordinates.
        """
        # gross transform to align knees
        self.gross_transform = self.meshes['femur'].rigidly_register(
            self.skm._mean_mesh[0], as_source=True, 
            apply_transform_to_mesh=False, return_transformed_mesh=False, 
            return_transform=True, max_n_iter=100, n_landmarks=1000, 
            reg_mode='similarity')

        self._bone_transforms = []
        point_coord_sets = []
        for bone_idx, bone in enumerate(MESH_PARAMS.keys()):
            mesh = self.meshes[bone].copy()
            mesh.apply_transform_to_mesh(self.gross_transform)

            # individual bone transform
            bone_transform = mesh.rigidly_register(
                self.skm._mean_mesh[bone_idx], as_source=True, 
                apply_transform_to_mesh=True, return_transformed_mesh=False, 
                return_transform=True, max_n_iter=100, n_landmarks=1000, 
                reg_mode='similarity')
            self._bone_transforms.append(bone_transform)
            
            # post-registration point coords
            point_coords = mesh.point_coords
            point_coord_sets.append(point_coords)

            if bone == 'femur':
                target_femur_point_weights = fade_mesh_points(point_coords,
                    fade_coord_thresh=FEMUR_Z_POINT_THRESH, fade_to='high')
            elif bone == 'tibia':
                target_tibia_point_weights = fade_mesh_points(point_coords,
                    fade_coord_thresh=TIBIA_Z_POINT_THRESH, fade_to='low')
            elif bone == 'patella':
                target_patella_point_weights = np.ones(point_coord_sets[2] \
                                                       .shape[0])
        
        self._target_coords = np.vstack(point_coord_sets)
        self._target_weights = concat_weights(target_femur_point_weights,
                                              target_tibia_point_weights, 
                                              target_patella_point_weights)
        
    def load(self) -> pd.DataFrame:
        """load the fitted params."""
        self.fitted_params = pd.read_csv(self.fitted_params_filename,
                                         index_col=0)
        self.meshes = get_meshes(os.path.join(self.sample_dir, 'mesh'), 
                                 self.right_knee, verbose=self.verbose)
        self.recon_meshes = get_meshes(self.recon_save_dir, self.right_knee,
                                       suffix='recon', verbose=self.verbose)
        if self.verbose:
            print('loaded fitted params and original and reconstructed',
                  f'meshes for {self.seg_filename}')
        return self.fitted_params

    def build_recon(self, pc_scores: torch.Tensor) -> torch.Tensor:
        """Build the reconstruction from the provided PC scores.
        
        Args:
            pc_scores: tensor (length n_pcs) of PC scores.

        Returns:
            recon_coords: coordinates of the reconstructed mesh.
        """
        pc_weights = pc_scores * torch.sqrt(self._Vs)
        pc_deformation = pc_weights @ self._PCs.T
        recon_coords = self._mean_coords \
            + pc_deformation * self._centered_coord_stds
        return recon_coords
    
    def fit(self, n_samples: int = N_SAMPLES_RECON, 
            min_n_epochs: int = MIN_N_EPOCHS, max_n_epochs: int = MAX_N_EPOCHS, 
            loss_thresh: float = LOSS_THRESH) -> bool:
        """Fit PC scores to the registered mesh. Returns False if no PCs were 
        fit (i.e., if crop ratio is not specified for the segmentation's 
        slice spacing).

        Args:
            min_n_epochs: minimum number of epochs to run. Defaults to 1000.
            max_n_epochs: maximum number of epochs to run. Defaults to 2000.
            ssm_loss_thresh: threshold for the SSM loss. Defaults to 0.01.

        """        
        if self.verbose:
            print(f'fitting for knee in {self.seg_filename}...')
        self.skm = StatisticalKneeModel()
        self.skm.load(SKM_DIR)
        
        self.meshes = get_meshes(self.seg_filename, self.right_knee, self.verbose)
        if not self.meshes:
            return False
        self.get_mesh_transforms_and_coords()

        self._PCs = torch.tensor(self.skm._PCs[:,:N_PCS], 
                                dtype=torch.float32)
        self._Vs = torch.tensor(self.skm._Vs[:N_PCS], 
                               dtype=torch.float32)
        self._mean_coords = torch.tensor(self.skm._coord_means, 
                                        dtype=torch.float32)
        self._centered_coord_stds = torch.tensor(self.skm._centered_coord_stds, 
                                                dtype=torch.float32)
        self._target_coords = torch.tensor(self._target_coords, 
                                          dtype=torch.float32).unsqueeze(0)
        self._target_weights = torch.tensor(self._target_weights, 
                                           dtype=torch.float32)
        
        self._PC_scores = torch.randn(N_PCS, requires_grad=True)
        optimizer = torch.optim.Adam([self._PC_scores], lr=0.01)
        
        optimization_complete = False
        epoch = 0
        while optimization_complete == False:
            optimizer.zero_grad()

            # Reconstruct using current scores
            recon_coords = self.build_recon(self._PC_scores)
            recon_coords = recon_coords.reshape(1, -1, 3)

            recon_weights = compute_point_weights(recon_coords[0], 
                                                  self.skm._mean_mesh)

            # resample
            n_target_coords = self._target_coords.shape[1]
            n_recon_coords = recon_coords.shape[1]
            target_idx = torch.randint(0, n_target_coords, (n_samples,))
            recon_idx = torch.randint(0, n_recon_coords, (n_samples,))

            target_coords = self._target_coords[0:1,target_idx,:]
            recon_coords = recon_coords[0:1,recon_idx,:]
            target_weights = self._target_weights[target_idx] # don't overwrite
            recon_weights = recon_weights[recon_idx]

            # take an optimization step
            loss = chamfer_distance(target_coords, recon_coords,
                                    x_point_weights=target_weights,
                                    y_point_weights=recon_weights)[0]
            loss.backward()
            optimizer.step()
        
            if epoch % 100 == 0:
                if self.verbose:
                    print(f'loss: {loss.item()}')
                self.log = pd.concat([self.log, 
                                      pd.DataFrame({'epoch': [epoch], 
                                                    'loss': [loss.item()]})
                                     ], ignore_index=True)
            if epoch > min_n_epochs and loss.item() < loss_thresh:
                optimization_complete = True
            if epoch > max_n_epochs:
                raise ValueError(f"""SSM fit loss {loss.item()} is greater 
                                 than threshold ({loss_thresh}).""")
            epoch += 1
        if self.verbose:
            print(f'Optimization complete. Loss: {loss.item()}')
        self.log = pd.concat([self.log,
                              pd.DataFrame({'epoch': [epoch], 
                                            'loss': [loss.item()]})
                             ], ignore_index=True)
            
        self.recon_coords = self.build_recon(self._PC_scores).detach().numpy()
        self.get_fitted_params()
        return True

    def save(self, override: bool = False):
        """Save the reconstructed mesh and intermediate meshes.
        
        Args:
            override: True to overwrite the fitted params. Defaults to False.
        """
        if not os.path.isfile(self.fitted_params_filename) or override==True:
            os.makedirs(self.recon_save_dir,
                        exist_ok=True)
            inverse_gross_transform = invert_transform(self.gross_transform)

            start_idx = 0
            self.recon_meshes = {}
            for bone_idx, bone in enumerate(MESH_PARAMS.keys()):
                bone_transform = self._bone_transforms[bone_idx]
                inverse_bone_transform = invert_transform(bone_transform)

                mean_mesh = self.skm._mean_mesh[bone_idx]
                n_points = mean_mesh.GetNumberOfPoints()
                end_idx = start_idx + n_points * 3
                recon_mesh = Mesh(create_vtk_mesh_from_deformed_points(
                    mean_mesh=mean_mesh, 
                    new_points=self.recon_coords[start_idx:end_idx]
                ))
                recon_mesh.save_mesh(os.path.join(self.recon_save_dir,
                    f'{bone}_recon_registered_to_mean.vtk'))
                
                self.recon_meshes[bone] = recon_mesh

                recon_mesh.apply_transform_to_mesh(inverse_bone_transform)
                recon_mesh.save_mesh(os.path.join(self.recon_save_dir,
                    f'{bone}_recon_registered_to_mean_femur.vtk'))

                recon_mesh.apply_transform_to_mesh(inverse_gross_transform)
                recon_mesh.save_mesh(os.path.join(self.recon_save_dir, 
                    f'{bone}_recon.vtk'))
                start_idx = end_idx 
            
            self.fitted_params.to_csv(self.fitted_params_filename)
            self.log.to_csv(os.path.join(self.recon_save_dir, 
                            f'loss_log_recon.csv'), index=False)
        else:
            raise Exception('fitted params previously saved. Exiting. You '
                            'may input override=True to overwrite.')

    def get_fitted_params(self):
        fitted_scores = self._get_pc_scores()
        fitted_transforms = self._get_transforms()
        self.fitted_params = pd.concat((fitted_transforms, fitted_scores)) \
            .rename('value').to_frame()

    def _get_pc_scores(self):
        """Return PC scores as a pandas series."""
        pc_scores = self._PC_scores.detach().numpy()
        pc_names = [f'pc{i+1}' for i in range(N_PCS)]
        pc_scores = pd.Series(pc_scores).rename(dict(zip(range(N_PCS), 
                                                         pc_names)))
        return pc_scores

    def _get_transforms(self):
        """Return tibia and patella transforms as a pandas series."""
        global_scale = pd.Series(get_transform_components(
            self.gross_transform, prefix='global_'))[['global_scale']]
        tibia_fitted_transforms = pd.Series(get_transform_components(
            self._bone_transforms[1], prefix='tibia_'))
        patella_fitted_transforms = pd.Series(get_transform_components(
            self._bone_transforms[2], prefix='patella_'))
        fitted_transforms = pd.concat((global_scale,
                                       tibia_fitted_transforms, 
                                       patella_fitted_transforms))
        return fitted_transforms
