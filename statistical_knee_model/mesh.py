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
import random
import string
import tempfile
from typing import Dict, Optional
import warnings

import numpy as np
import pyvista as pv
import vtk
import SimpleITK as sitk

import pymskt.image as msktimage
from pymskt.mesh import BoneMesh
from pymskt.mesh.meshTransform import (
    create_transform, 
    apply_transform
)
from pymskt.utils import safely_delete_tmp_file

from constants import MESH_PARAMS


SITK_SAGITTAL_AXIS = 2
ISOTROPIC_SPACING = 1.1
SMOOTH_IMAGE_VAR = 1.0
MARCHING_CUBES_THRESHOLD = 0.5

loc_tmp_save = tempfile.gettempdir()
tmp_filename = ''.join(random.choice(string.ascii_lowercase) 
                       for i in range(10)) + '.nrrd'


def flip_seg_laterality(segpath: str, 
                        sitk_slice_axis_idx: int = SITK_SAGITTAL_AXIS) -> None:
    """Flip the laterality (left/right) of the segmentation.
    
    Args:
        segpath: path to the segmentation.
        np_slice_axis_idx: index representing the slice axis when in numpy 
            array form.

    Returns:
        segpath: path to the new right knee segmentation.
    """
    seg = sitk.ReadImage(segpath)
    flip_axes = [True if i == sitk_slice_axis_idx else False for i in range(seg.GetDimension())]
    seg_flipped = sitk.Flip(seg, flip_axes, flipAboutOrigin=True)

    segpath = segpath[:segpath.find('.nrrd')] + '_RIGHT.nrrd'
    sitk.WriteImage(seg_flipped, segpath)
    return segpath


def set_2d_image_border_to_zeros(image: sitk.Image, border_size: int = 1
                                 ) -> sitk.Image:
    """Utility function to ensure that all segmentations are "closed" for 
    identifying isolines. If the segmentation extends to the edges of the 
    image then the surface won't be closed at the places it touches the edges.
    Based on pymskt.image.main.set_seg_border_to_zeros, which is appropriate 
    for 3D segmentations.

    Args:
        image: 2D
        border_size: the size of the border to set around the edges of the 2D 
            image, by default 1.

    Returns:
        new_image: the image with border set to 0 (background).
    """    
    image_array = sitk.GetArrayFromImage(image)
    new_image_array = np.zeros_like(image_array)
    new_image_array[border_size:-border_size, border_size:-border_size] \
        = image_array[border_size:-border_size, border_size:-border_size]
    new_image = sitk.GetImageFromArray(new_image_array)
    new_image.CopyInformation(image)
    return new_image


class BoneMeshNonIso(BoneMesh):
    """Class to create, store, and process nonisotropic bone meshes. This 
    class is a modified version of the PyMSKT BoneMesh class."""
    def __init__(self, mesh=None, seg_image=None, path_seg_image=None, 
                 label_idx=None, min_n_pixels=5000, crop_percent=None,
                 bone='femur',):
        """Class initialization"""
        self._crop_percent = crop_percent
        self._bone = bone
        super().__init__(mesh=mesh,
                         seg_image=seg_image,
                         path_seg_image=path_seg_image,
                         label_idx=label_idx,
                         min_n_pixels=min_n_pixels,
                         crop_percent=crop_percent,
                         bone=bone)
        
    def create_2d_mesh(self, slice_image: np.ndarray, 
        smooth_image: bool = True, smooth_image_var: float = SMOOTH_IMAGE_VAR,
        marching_cubes_threshold: float = MARCHING_CUBES_THRESHOLD
        ) -> pv.PolyData:
        """create a 2D mesh using the continuous marching cubes algorithm.
        """
        slice_image = set_2d_image_border_to_zeros(slice_image)
        if smooth_image:
            slice_image = msktimage.smooth_image(slice_image, self.label_idx, 
                                                 smooth_image_var)
        # set 2D image info
        slice_image.SetOrigin((0,0))
        slice_image.SetDirection((1,0,0,1)) # rotation matrix - identity
        slice_image.SetSpacing((1,1))

        # write temp file so we can read with VTK
        sitk.WriteImage(slice_image, os.path.join(loc_tmp_save, tmp_filename))
        vtk_image_reader = msktimage.read_nrrd(os.path.join(loc_tmp_save, 
                                tmp_filename),set_origin_zero=True)
        
        # continuous marching squares
        mc = vtk.vtkMarchingContourFilter()
        mc.SetInputConnection(vtk_image_reader.GetOutputPort())
        mc.ComputeNormalsOn()
        mc.ComputeGradientsOn()
        mc.SetValue(0, marching_cubes_threshold)
        mc.Update()
        slice_mesh = pv.PolyData(mc.GetOutput())
        return slice_mesh
    
    def create_mesh(self, smooth_image: bool = True, 
                    smooth_image_var: float = SMOOTH_IMAGE_VAR, 
                    marching_cubes_threshold: float = MARCHING_CUBES_THRESHOLD,
                    label_idx: Optional[int] = None,
                    min_n_pixels: Optional[int] = None,
                    crop_percent: Optional[float] = None):
        if label_idx is not None:
            self._label_idx = label_idx
        if min_n_pixels is not None:
            self._min_n_pixels = min_n_pixels
        if crop_percent is not None:
            self._crop_percent = crop_percent
        if self._seg_image is None:
            self.read_seg_image()

        seg_image = self._crop_mesh()

        n_slices = seg_image.GetSize()[SITK_SAGITTAL_AXIS]

        seg_view = sitk.GetArrayViewFromImage(seg_image)
        n_pixels_labelled = sum(seg_view[seg_view == self.label_idx])
        if n_pixels_labelled < self.min_n_pixels:
            raise Exception(f"""The mesh does not exist in this segmentation! 
                            Only {n_pixels_labelled} pixels detected, 
                            threshold # is {marching_cubes_threshold}""")
        
        # append 2D slice meshes
        appendFilter = vtk.vtkAppendPolyData()
        for slice_idx in range(n_slices):
            slice_image = seg_image[:,:,slice_idx] # for sagittal slices
            slice_mesh = self.create_2d_mesh(slice_image, 
                            smooth_image=smooth_image, 
                            smooth_image_var=smooth_image_var,
                            marching_cubes_threshold=marching_cubes_threshold)
            slice_mesh.points[:,2] = slice_idx
            appendFilter.AddInputData(slice_mesh)
        appendFilter.Update()
        full_mesh = pv.PolyData(appendFilter.GetOutput())

        # copy image transform to mesh
        full_mesh.points = full_mesh.points * seg_image.GetSpacing()
        image_direction = np.reshape(seg_image.GetDirection(), (3,3))
        transform_array = np.identity(4)
        transform_array[:3,:3] = image_direction
        transform_array[:3,3] = seg_image.GetOrigin()
        transform = create_transform(transform_array)
        mesh_ = apply_transform(source=full_mesh, transform=transform)
        self.deep_copy(mesh_)

        self.load_mesh_scalars()
        safely_delete_tmp_file(tempfile.gettempdir(),
                               tmp_filename)

    def _crop_mesh(self):
        """if mesh is of femur or tibia, crop long axis with crop ratio."""
        if self.crop_percent is not None and (('femur' in self._bone) 
                                              or ('tibia' in self._bone)):
            if 'femur' in self.bone:
                bone_crop_distal = True
            elif 'tibia' in self.bone:
                bone_crop_distal = False
            else:
                raise Exception(f"""var bone should be "femur" or "tibia" 
                                got: {self._bone} instead""")

            seg_image = msktimage.crop_bone_based_on_width(self.seg_image, 
                            self.label_idx, 
                            percent_width_to_crop_height = self.crop_percent, 
                            bone_crop_distal=bone_crop_distal)
            
        elif self.crop_percent is not None:
            warnings.warn(f"""Trying to crop bone, but {self._bone} specified 
                          and only bones `femur` or `tibia` currently 
                          supported for cropping. If using another bone, 
                          consider making a pull request. If cropping not 
                          desired, set `crop_percent=None`.""")
        else:
            seg_image = self.seg_image
        return seg_image


def create_meshes(segpath: str, right_knee: bool, verbose: bool = False
                  ) -> str:
    """Create the bone meshes from a segmentation file.
    
    Args:
        segpath: path containing the segmentation from which to create the 
            meshes.
        right_knee: whether the segmentation is for a right knee.
        verbose: whether to print messages about the process.

    Returns:
        meshes: dictionary containing meshes corresponding to each bone.
    """
    mesh_dir = os.path.join(os.path.split(os.path.dirname(segpath))[0], 
                            'mesh')
    os.makedirs(mesh_dir, exist_ok=True)
    meshes = {}
    if not right_knee:
        segpath = flip_seg_laterality(segpath)

    seg = sitk.ReadImage(segpath)
    spacing_between_slices = seg.GetSpacing()[SITK_SAGITTAL_AXIS]
    if spacing_between_slices <= ISOTROPIC_SPACING:
        crop_ratio_key = 'iso'
    else:
        crop_ratio_key = f'{round(spacing_between_slices, 1)}mm'
        
    for bone_name, bone_params in MESH_PARAMS.items():
        if crop_ratio_key not in bone_params['crop_ratios']:
            raise ValueError(f"""Volumes with {spacing_between_slices} 
                                spacing between slices not supported. At this 
                                time, only isotropic volumes and volumes with 
                                3.3, 3.5, and 4.0mm spacing between slices are 
                                supported.""")
        
        if crop_ratio_key == 'iso':
            # create mesh with typical pymskt workflow
            bone_mesh = BoneMesh(path_seg_image=segpath,
                label_idx=bone_params['label_idx'],
                crop_percent=bone_params['crop_ratios'][crop_ratio_key],
                bone=bone_name)
            bone_mesh.create_mesh(smooth_image_var=SMOOTH_IMAGE_VAR)
            bone_mesh.resample_surface(clusters=bone_params['n_pts'])
        else:
            # create mesh with vertices that only exist on image slice planes
            bone_mesh = BoneMeshNonIso(path_seg_image=segpath,
                label_idx=bone_params['label_idx'], 
                crop_percent=bone_params['crop_ratios'][crop_ratio_key], 
                bone=bone_name)
            bone_mesh.create_mesh(smooth_image_var=SMOOTH_IMAGE_VAR)
        meshes[bone_name] = bone_mesh
        bone_mesh.save_mesh(os.path.join(mesh_dir, f'{bone_name}.vtk'))
    return meshes


def get_meshes(filepath: str, right_knee: bool, suffix: str = '',
               verbose: bool = False) -> Dict[str, vtk.vtkPolyData]:
    """get the bone meshes.
    
    Args:
        filepath: segmentation file path or path containing meshes.
        right_knee: whether the segmentation is for a right knee.
        suffix: suffix to add to the mesh file names.
        verbose: whether to print messages about the process.

    Returns:
        meshes: dictionary containing meshes corresponding to each bone.
    """
    if os.path.isfile(filepath):
        file_dir = os.path.dirname(filepath)
    else:
        file_dir = filepath
    meshes = {}
    for bone in MESH_PARAMS.keys():
        if suffix:
            filename = f'{bone}_{suffix}.vtk'
        else:
            filename = f'{bone}.vtk'
        meshpath = os.path.join(file_dir, filename)
        if os.path.exists(meshpath):
            meshes[bone] = BoneMesh(mesh=meshpath)
        else:
            meshes = create_meshes(filepath, right_knee, verbose)
            break
    return meshes
