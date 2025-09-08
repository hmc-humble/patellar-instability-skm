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
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

torch.manual_seed(0)


class UNet(nn.Module):
    """
    Example usage:
    >>> model = UNet(classes=4)
    """
    def __init__(self, in_channels=1, classes=1, kernel_size=3, 
                 dropout_rate=0):
        super().__init__()
        self.layers = [in_channels, 64, 128, 256, 512, 1024]
        self.double_conv_downs = nn.ModuleList(
            [self.__double_conv(layer, next_layer, kernel_size, dropout_rate)
             for layer, next_layer in zip(self.layers[:-1], self.layers[1:])])
        self.up_transpose = nn.ModuleList(
            [nn.ConvTranspose2d(layer, next_layer, kernel_size=2, stride=2)
             for layer, next_layer in zip(self.layers[::-1][:-2], 
                                          self.layers[::-1][1:-1])])
        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer, layer//2, kernel_size, dropout_rate)
             for layer in self.layers[::-1][:-2]])
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=dropout_rate)
        # dropout implemented according to:
        # https://www.kaggle.com/code/phoenigs/u-net-dropout-
        # augmentation-stratification/notebook
        self.final_conv = nn.Conv2d(self.layers[1], classes, kernel_size=1)
        self.softmax = nn.Softmax2d()

        
    def __double_conv(self, in_channels, out_channels, kernel_size, 
                      dropout_rate):
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                      padding='same'), 
            nn.BatchNorm2d(out_channels),
            nn.Dropout(p=dropout_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                      padding='same'),
            nn.ReLU(inplace=True)
        )
        return conv
    
    
    def forward(self, x):
        # down layers
        concat_layers = []
        
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                x = self.max_pool(x)
                x = self.dropout(x)
        
        concat_layers = concat_layers[::-1]
        
        # up layers
        for up_transpose, double_conv_up, concat_layer in zip(self.up_transpose, self.double_conv_ups, concat_layers):
            x = up_transpose(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layers.shape[2:])
            
            concatenated = torch.cat((concat_layer, x), dim=1)
            concatenated = self.dropout(concatenated)
            x = double_conv_up(concatenated)
        
        x = self.final_conv(x)
        x = self.softmax(x)
        return x
