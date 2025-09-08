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
"""
Mesh parameters are included below. Nominally, the femur is cropped such that 
the height is 1.05 times the width, the tibia is cropped such that the height 
is 0.7 times the width, and the patella is not cropped. This only applies to 
isotropic (3D) volumes. If the volume has spacing between slices of 3.3, 3.5, 
or 4.0 mm, the crop ratios are adjusted based on an analysis thay compared 
bone widths as measured on isotropic and nonisotropic volumes.

The number of points to sample from the mesh is also included (the femur and 
tibia surfaces are larger than the patella surface, so more points are 
sampled).
"""
MESH_PARAMS = {
    "femur": {
        "crop_ratios": {
            "iso": 1.05, # original isotropic segmentation crop ratio
            "3.3mm": 1.069715950978765,
            "3.5mm": 1.1039940496068994,
            "4.0mm": 1.1024937179577505
        },
        "n_pts": 10000,
        "label_idx": 1
        },
    "tibia": {
        "crop_ratios": {
            "iso": 0.7,
            "3.3mm": 0.7311816947920154,
            "3.5mm": 0.7466671074751916,
            "4.0mm": 0.7436361428487898,
        },
        "n_pts": 10000,
        "label_idx": 3
        },
    "patella": {
        "crop_ratios": {
            "iso": None,
            "3.3mm": None,
            "3.5mm": None,
            "4.0mm": None,
        },
        "n_pts": 3000,
        "label_idx": 2
        }
    }

FEMUR_Z_POINT_THRESH = 30
TIBIA_Z_POINT_THRESH = -60
