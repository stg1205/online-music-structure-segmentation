import torch
from torch import nn
import sys
sys.path.append('..')
from utils.modules import MyConv2d
from utils.modules import MyDense

class UnsupEmbedding(nn.Module):
    '''
    Model from the paper "Unsupervised learning of deep features for music segmentation"
    Three Conv2d -> three dense layers    
    replace the kernel with (3, 3) and padding with (2, 2)
    '''
    def __init__(self, input_shape, channels=64):
        super(UnsupEmbedding, self).__init__()
        # (batch, channel, f_bin, time_bin) = B， 1， 80， 64

        # self._conv1 = MyConv2d(1, channels, kernel_size=(6, 4), pooling=(2, 4))  # 
        # self._conv2 = MyConv2d(channels, channels*2, (6, 4), (3, 4))
        # self._conv3 = MyConv2d(channels*2, channels*4, (6, 4), (2, 4))
        
        self._conv1 = MyConv2d(1, channels)  # B， channels， 40， 32
        self._conv2 = MyConv2d(channels, channels*2)  # B， 2*channels， 20， 16
        self._conv3 = MyConv2d(channels*2, channels*4)  # B， 4*channels， 10, 8

        dense_in = int(4*channels * input_shape[0]/2**3 * input_shape[1]/2**3)
        
        self._dense1 = MyDense(dense_in, 128)
        self._dense2 = MyDense(128, 128)
        self._dense3 = MyDense(128, 128)
    
    def forward(self, x):
        x = x.unsqueeze(1)  # channel dimension
        x = self._conv1(x)
        x = self._conv2(x)
        x = self._conv3(x)
        
        x = x.reshape(x.size(0), -1)
        x = self._dense1(x)
        x = self._dense2(x)
        out = self._dense3(x)

        return out


class FrontEnd(nn.Module):
    '''Supervised metric learning model.
    Also the front-end of the policy model to output features.
    '''
    def __init__(self, input_shape) -> None:
        super(FrontEnd, self).__init__()

    
    def forward(self):
        pass
