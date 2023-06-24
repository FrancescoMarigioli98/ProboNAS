# Script that given a genetic code creates the Net.

import torch.nn as nn
from invbneckzpadd import InvBNeck
from convnextzpadd import ConvNext
from Downsampling import Downsampling
from torch import flatten
from collections import OrderedDict

# Function that creates a single block.
def CreateBlock(block_list, in_channels, resolution, is_training=False):
    blockType=block_list[0]
    out_channels=block_list[1]
    kernel_gene=block_list[2] 
    exp_ratio=block_list[3]
    
    if blockType=='i':
        block=InvBNeck(in_channels=in_channels, out_channels=out_channels, 
                       kernel=kernel_gene*2+1, channel_factor=exp_ratio, is_training=is_training)
    elif blockType=='c':
        block=ConvNext(in_channels=in_channels, out_channels=out_channels, 
                       kernel=kernel_gene*2+3, channel_factor=exp_ratio, resolution=resolution, is_training=is_training)
    else:
        raise Exception("Block not recognized.\n") 
    
    return block

def CreateDict(stagenum, stage_list, in_channels, resolution, is_training=False):
      
      listDict=[]
      for i in range(len(stage_list[stagenum])):
        if i==0:
          if stagenum==0:
            tupl=('block'+str(i),CreateBlock(stage_list[stagenum][i], in_channels=in_channels, resolution=resolution, is_training=is_training))
          else:
            tupl=('block'+str(i),CreateBlock(stage_list[stagenum][i], in_channels=stage_list[stagenum-1][len(stage_list[stagenum-1])-1][1], resolution=resolution, is_training=is_training))
        else:
          tupl=('block'+str(i),CreateBlock(stage_list[stagenum][i], in_channels=stage_list[stagenum][i-1][1], resolution=resolution, is_training=is_training))

        listDict.append(tupl)
      return listDict

# Class that defines our Net.
class Net(nn.Module):
    def __init__(self, in_channels, classes, stage_list, is_training=False):
        super(Net, self).__init__()
        
        stem_channels=stage_list[0]
        stage_list=stage_list[1]

        self.resolution=128

        self.stem = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=stem_channels, kernel_size=2, stride=2),
                                  nn.GELU(),
                                  nn.BatchNorm2d(num_features=stem_channels))
        
        self.resolution=self.resolution//2

        in_channels=stem_channels

        # STAGE 0
        self.stage0=nn.Sequential(OrderedDict(CreateDict(0, stage_list, in_channels, self.resolution, is_training=is_training)))
        
        self.ds0=Downsampling(channels=stage_list[0][len(stage_list[0])-1][1])
        self.resolution=self.resolution//2
        
        # STAGE 1
        self.stage1=nn.Sequential(OrderedDict(CreateDict(1, stage_list, in_channels, self.resolution, is_training=is_training)))
        
        self.ds1=Downsampling(channels=stage_list[1][len(stage_list[1])-1][1])
        self.resolution=self.resolution//2
        
        # STAGE 2
        self.stage2=nn.Sequential(OrderedDict(CreateDict(2, stage_list, in_channels, self.resolution, is_training=is_training)))
        
        self.ds2=Downsampling(channels=stage_list[2][len(stage_list[2])-1][1])
        self.resolution=self.resolution//2
        
        # STAGE 3
        self.stage3=nn.Sequential(OrderedDict(CreateDict(3, stage_list, in_channels, self.resolution, is_training=is_training)))
        
        self.ds3=Downsampling(channels=stage_list[3][len(stage_list[3])-1][1])
        self.resolution=self.resolution//2

        
        # OUTPUT       
        self.fc=nn.Linear(in_features=stage_list[3][len(stage_list[3])-1][1]*(self.resolution**2), out_features=classes)        
        
    def forward(self, x):
        x=self.stem(x)
        x=self.stage0(x)
        x=self.ds0(x)
        x=self.stage1(x)
        x=self.ds1(x)
        x=self.stage2(x)
        x=self.ds2(x)
        x=self.stage3(x)
        x=self.ds3(x)
        x = flatten(x, 1)
        x=self.fc(x)
        return x
    
