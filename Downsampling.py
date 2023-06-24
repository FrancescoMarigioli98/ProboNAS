# Class that implements the Downsampling Block.

from torch import nn

class Downsampling(nn.Module):
  def __init__(self, channels):
    super(Downsampling, self).__init__()

    self.dw=nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=3, padding=1, stride=2, groups=channels)
    self.pw=nn.Conv2d(in_channels=channels, out_channels=channels,
                       kernel_size=1)
    
    self.bn=nn.BatchNorm2d(num_features=channels)
    self.relu=nn.ReLU()
    
  def forward(self, x):
    y=self.dw(x)
    y=self.bn(y)

    z=y
    y=self.pw(y)
    y=self.bn(y)
    y=self.relu(y)

    return y