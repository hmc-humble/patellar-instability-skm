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
from typing import Dict

import pymskt as mskt
from scipy.spatial.transform import Rotation
import vtk


def get_transform_components(transform: vtk.vtkTransform, prefix: str = '', 
                             verbose=False) -> Dict[str, float]:
    """return the scale, rotation, and translation components of a transform in 
    the form of a dictionary.

    Args:
        transform: transform to decompose.
        prefix: optional prefix to the transform component names. Defaults to ''.
        verbose: optional flag to print the transform components. Defaults to 
            False.
    """
    T = mskt.mesh.meshTransform.get_linear_transform_matrix(transform)
    scale, unit_transform = mskt.mesh.meshTransform.separate_scale_and_transform(T)
    r = Rotation.from_matrix(unit_transform)
    rotations = r.as_euler('xyz', degrees=True) # euler angles
    translations = T[:3, 3]

    if verbose:
        print(f'Scale: {scale}')
        print(f'Rotations: {rotations}')
        print(f'Translations: {translations}')

    transform_components = {f'{prefix}scale': scale,
                            f'{prefix}x': translations[0],
                            f'{prefix}y': translations[1],
                            f'{prefix}z': translations[2],
                            f'{prefix}rx': rotations[0],
                            f'{prefix}ry': rotations[1],
                            f'{prefix}rz': rotations[2],
                            }
    return transform_components


def invert_transform(transform: vtk.vtkTransform):
    """Return a new inverted transform object (rather than over-
    writing the original transform object with its inverse).

    Args:
        transform: original transform to invert.

    Returns:
        inverse_transform: a separate inverted transform object.
    """
    inverse_transform = vtk.vtkTransform()
    inverse_transform.SetMatrix(transform.GetMatrix())
    inverse_transform.Inverse()
    return inverse_transform
