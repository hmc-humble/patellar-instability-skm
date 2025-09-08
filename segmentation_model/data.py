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
import numpy as np
import cv2
import SimpleITK as sitk


def get_series(series_dir, return_info=False):
    """Read in a series of dicom images and return the series as a numpy 
    array. Optionally return the origin, direction, and spacing of the 
    series.
    
    Args:
        series_dir (str): path to the directory containing the dicom files
        return_info (bool): if True, return the origin, direction, and 
            spacing of the series
    """
    series_reader = sitk.ImageSeriesReader()
    series_sitk = sitk.ReadImage(
        series_reader.GetGDCMSeriesFileNames(series_dir))
    series = sitk.GetArrayFromImage(series_sitk)
            
    if return_info:
        info = {'origin': series_sitk.GetOrigin(),
                'direction': series_sitk.GetDirection(),
                'spacing': series_sitk.GetSpacing()
               }
        return series, info
    return series

def resize_series(series, interpolation, new_size):
    """Resize each slice in a series of images to a new size.

    Args:
        series (np.ndarray): series of images.
        interpolation (int): interpolation method to use.
        new_size (list): new size of the series.

    Returns:
        np.ndarray: resized series of images.
    """
    n_slices = len(series)
    series_rs = np.zeros((n_slices, new_size[0], new_size[1]))
    for slice_idx in range(n_slices):
        series_rs[slice_idx] = cv2.resize(series[slice_idx], 
                                          dsize=new_size[::-1], 
                                          interpolation=interpolation)
    return series_rs
    
    
def scale_min_max(series):
    """Scale the values of a series of images to be between 0 and 1.

    Args:
        series (np.ndarray): series of images.

    Returns:
        np.ndarray: series of images with values scaled between 0 and 1.
    """    
    min_val = np.min(series)
    series = series-min_val
    max_val = np.max(series)
    series = series/max_val
    return series
