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
import cv2
import numpy as np
import torch
import SimpleITK as sitk
from skimage.measure import label

from model import UNet
from data import get_series, resize_series, scale_min_max

torch.manual_seed(0)
np.random.seed(0)


MODEL_DIR = os.path.join(os.path.dirname(__file__), 'weights')

CLASS_LABELS = {'background': 0, 'femur': 1, 'patella': 2, 'tibia': 3}
N_LABELS = len(CLASS_LABELS)

INPUT_SIZE = [512, 512]
INTERPOLATION_INPUT = cv2.INTER_CUBIC
INTERPOLATION_PREDICTION = cv2.INTER_NEAREST
KERNEL_SIZE = 5


def predict_seg(model, series):
    """Predict the hard label segmentation on a series of images.

    Args:
        model (torch.nn.Module): trained model.
        series (np.ndarray): series of images.

    Returns:
        np.ndarray: softmax output of the model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    series_resized = resize_series(series, interpolation=INTERPOLATION_INPUT, 
                                   new_size=INPUT_SIZE)
    series_input = scale_min_max(series_resized)
    
    predictions_shape = (series.shape[0], N_LABELS, 
                         series.shape[1], series.shape[2])
    predictions = np.zeros(predictions_shape)
    for i in range(len(series)):
        image = series_input[i]
        image = image[np.newaxis,np.newaxis,:,:]
        image = torch.from_numpy(image).type(torch.float32).to(device)
        prediction = model(image).to('cpu').detach().numpy()
        prediction = resize_series(prediction[0], 
                                   interpolation=INTERPOLATION_PREDICTION, 
                                   new_size=series.shape[1:])
        predictions[i] = prediction
    
    n_classes = predictions.shape[1]
    hard_labels = predictions.copy()
    for class_ in range(n_classes):
        hard_labels[:,class_,:,:] = (class_==np.argmax(hard_labels, axis=1))
    return hard_labels


def clean_seg(volume):
    """Clean a segmented volume by keeping the largest connected component 
    (CC) for each class.

    Args:
        volume (np.ndarray): segmented volume.
    
    Returns:
        np.ndarray: cleaned segmented volume.
    """
    # keep track of all non-background pixels
    n_sag_slices = volume.shape[0]
    n_ax_slices = volume.shape[2]
    n_cor_slices = volume.shape[3]
    
    volume_clean = volume.copy()
    new_segmented_pixels = np.zeros((n_sag_slices, n_ax_slices, n_cor_slices))
    for class_ in range(1, N_LABELS): # start at 1 to exclude background
        volume_clean[:,class_,:,:] = keep_largest_ccs(volume_clean[:,class_,:,:])
        new_segmented_pixels += volume_clean[:,class_,:,:]
    
    volume_clean[:,0,:,:] = (new_segmented_pixels==0) # set background
    return volume_clean


def keep_largest_ccs(volume):
    """Keep the largest connected component of a 3D binary volume.

    Args:
        volume (np.ndarray): 3D binary volume.

    Returns:
        np.ndarray: 3D binary volume with only the largest connected 
            component.
    """
    cc_labels = label(volume, connectivity=1)
    if cc_labels.max() != 0:
        largest_cc_label = np.argmax(np.bincount(cc_labels.flat)[1:])+1
        largest_cc = (cc_labels==largest_cc_label)
        volume = largest_cc.astype(int)
    return volume


def segment_image(series_dir, output_filename):
    """Segment a single image.
    
    Args:
        series-dir (str): path to DICOM series directory.
        output_filename (str): path of output filename.

    Returns:
        None
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    series, info = get_series(series_dir, return_info=True)
    # load model
    model_savefile = os.path.join(MODEL_DIR, f'weights_sagittal_view.pt')
    if not os.path.exists(model_savefile):
        raise FileNotFoundError(''.join(['Model file not found: ', 
                f'{model_savefile}. Please download from ',
                'https://drive.google.com/file/d/1Edn2ZWCvjQttj8sAsG302HOmrl',
                '17ySfB/view?usp=drive_link']))
    model = UNet(in_channels=1, classes=4, kernel_size=KERNEL_SIZE
                    ).to(device)
    model.load_state_dict(torch.load(model_savefile, weights_only=False))

    # predict
    seg = predict_seg(model, series)
    seg = clean_seg(seg)

    # decode from one-hot encoding
    seg = np.argmax(seg, axis=1)

    # save
    seg = seg.astype('float64')
    seg = sitk.GetImageFromArray(seg)
    seg.SetOrigin(info['origin'])
    seg.SetDirection(info['direction'])
    seg.SetSpacing(info['spacing'])

    sitk.WriteImage(seg, output_filename, useCompression=True)
