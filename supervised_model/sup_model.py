import torch
from torch import nn


class FrontEnd(nn.Module):
    '''Supervised metric learning model.
    Also the front-end of the policy model to output features.
    '''
    def __init__(self, input_shape) -> None:
        super(FrontEnd, self).__init__()
    
    def forward(self):
        pass
