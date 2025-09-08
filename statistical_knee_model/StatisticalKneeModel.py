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

import numpy as np

import pymskt as mskt


class StatisticalKneeModel():
    def __init__(self, weight_coords: bool = True) -> None:
        """Initialize the shape model object.
        
        Args:
            weight_coords: True to weight the coordinates. Defaults to True.
        """
        self.weight_coords = weight_coords

    def load(self, load_dir: str) -> None:
        """Load the statistical knee model.

        Args:
            load_dir: directory from which to load the statistical knee model.
        """
        self._coord_means = np.load(os.path.join(load_dir, 'coord_means.npy'))
        self._centered_coord_stds = np.load(
            os.path.join(load_dir, 'centered_coord_stds.npy'))
        self._PCs = np.load(os.path.join(load_dir, 'PCs.npy'))
        self._Vs = np.load(os.path.join(load_dir, 'Vs.npy'))
        
        self._mean_mesh = []
        mean_mesh_dir = os.path.join(load_dir, 'mean_mesh')
        mean_mesh_files = sorted(os.listdir(mean_mesh_dir))
        for mesh_file in mean_mesh_files:
            if '.vtk' in mesh_file:
                mesh = mskt.mesh.io.read_vtk(os.path.join(mean_mesh_dir, mesh_file))
                self._mean_mesh.append(mesh)
