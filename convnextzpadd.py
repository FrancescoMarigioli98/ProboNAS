# Class that implements the ConvNeXt Block.

import torch
from torch import nn
import torch.nn.functional as F


class ConvNext(nn.Module):
  def __init__(self, in_channels, out_channels, kernel=3, channel_factor=2, resolution=224, is_training=False):
    super(ConvNext, self).__init__()

    self.dw=nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                      kernel_size=(kernel,kernel), padding=(kernel-1)//2,
                      groups=in_channels)
    self.pw1=nn.Conv2d(in_channels=in_channels, out_channels=out_channels*channel_factor,
                       kernel_size=(1,1) )
    self.pw2=nn.Conv2d(in_channels=out_channels*channel_factor, out_channels=out_channels,
                       kernel_size=(1,1))
    self.ln=LayerNorm(in_channels, eps=1e-6)
    self.gelu=nn.GELU()

    self.is_training=is_training
    
  def forward(self, x):
    y=self.dw(x)
    y=self.ln(y)

    y=self.pw1(y)
    y=self.gelu(y)

    y=self.pw2(y)



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


class LayerNorm(nn.Module):
  def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
      super().__init__()
      self.weight = nn.Parameter(torch.ones(normalized_shape))
      self.bias = nn.Parameter(torch.zeros(normalized_shape))
      self.eps = eps
      self.data_format = data_format
      if self.data_format not in ["channels_last", "channels_first"]:
          raise NotImplementedError 
      self.normalized_shape = (normalized_shape, )
  
  def forward(self, x):
      if self.data_format == "channels_last":
          return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
      elif self.data_format == "channels_first":
          u = x.mean(1, keepdim=True)
          s = (x - u).pow(2).mean(1, keepdim=True)
          x = (x - u) / torch.sqrt(s + self.eps)
          x = self.weight[:, None, None] * x + self.bias[:, None, None]
          return x