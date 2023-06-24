# Class that implements the MobileNet Block.

import torch
from torch import nn

class InvBNeck(nn.Module):
  def __init__(self, in_channels, out_channels, kernel=3, channel_factor=2, resolution=224, is_training=False):
    super(InvBNeck, self).__init__()

    self.pw1=nn.Conv2d(in_channels=in_channels, out_channels=out_channels*channel_factor,
                       kernel_size=(1,1) )
    self.dw=nn.Conv2d(in_channels=out_channels*channel_factor, out_channels=out_channels*channel_factor,
                      kernel_size=(kernel,kernel), padding=(kernel-1)//2,
                      groups=out_channels*channel_factor)
    self.pw2=nn.Conv2d(in_channels=out_channels*channel_factor, out_channels=out_channels,
                       kernel_size=(1,1))
    self.bn1=nn.BatchNorm2d(num_features=out_channels*channel_factor)
    self.bn2=nn.BatchNorm2d(num_features=out_channels*channel_factor)
    self.bn3=nn.BatchNorm2d(num_features=out_channels)
    self.relu=nn.ReLU()

    self.is_training=is_training
    
  def forward(self, x):
    y=self.pw1(x)
    y=self.bn1(y)
    y=self.relu(y)

    y=self.dw(y)
    y=self.bn2(y)
    y=self.relu(y)

    y=self.pw2(y)
    y=self.bn3(y)

    #ZERO PADDING
    shortcut=x
    featuremap_size = y.size()[2:4]
    batch_size = y.size()[0]
    residual_channel = y.size()[1]
    shortcut_channel = shortcut.size()[1]

    if residual_channel != shortcut_channel:
        padding = torch.autograd.Variable(torch.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0], featuremap_size[1]).fill_(0)) 
        if(self.is_training):
          padding=padding.to(torch.device("cuda:0"))
        y += torch.cat((shortcut, padding), 1)
    else:
        y += shortcut 

    return y