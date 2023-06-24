# Script that implements ProboNAS, our evolutionary algorithm.

import torch
import torchvision
from torch.nn import Module
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import MaxPool2d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten
from torchvision import transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset

from pyvww.pytorch import VisualWakeWordsClassification

import numpy as np
import pandas as pd

from invbneckzpadd import InvBNeck

from convnextzpadd import ConvNext

from Downsampling import Downsampling

from createNetStem import Net
from measures import *
from measures.logsynflow import compute_synflow_per_weight
from measures.naswot import compute_naswot_score
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import parameter_count

from randomNetStem import randomGen
import random
import time

from PIL import Image

# Function that checks if the output channels are in the right interval
def check_out_channels(genetic_code):
  stem_channels, genetic_code = genetic_code
  if(stem_channels>10):
    stem_channels=10
  if(stem_channels<3):
    stem_channels=3
  
  prev_out=stem_channels
  max_channels=20
  min_channels=10
  for stage in genetic_code:
    for block in stage:
      if(block[1]>max_channels):
        block[1]=max_channels
      if(block[1]<min_channels):
        block[1]=min_channels

      if(block[1]<prev_out):
        block[1]=prev_out
      else:
        prev_out=block[1]
    min_channels=max_channels
    max_channels=max_channels*2
  return (stem_channels, genetic_code)

# Function that mutates a single exemplar
def mutation(genetic_code, params):
  stem_channels, genetic_code = genetic_code
  p_block=params["p_block"]
  p_gene=params["p_gene"]
  std_channels=params["std_channels"]

  if(np.random.rand()<p_gene):
    stem_channels=int(np.random.normal(stem_channels, std_channels)+0.5)

  for stage in genetic_code:
    for block in stage:
      if(np.random.rand()<p_block):
        if(np.random.rand()<p_gene):  
          if(block[0]=='c'):
            block[0]='i'
          else:
            block[0]='c'
        if(np.random.rand()<p_gene): 
          block[1]=int(np.random.normal(block[1], std_channels)+0.5)
        if(np.random.rand()<p_gene): 
          block[2]=np.random.randint(0,4)
        if(np.random.rand()<p_gene):
          block[3]=np.random.randint(2,5)
  
  return check_out_channels((stem_channels, genetic_code))

# Function that creates a new exemplar crossgenearting from two parent exemplars
def cross(mother, father):
  genetic_code=[]
  stem_m=mother[0]
  mother=mother[1]
  
  stem_f=father[0]
  father=father[1]

  if(np.random.rand()<0.5):
    stem=stem_m
  else:
    stem=stem_f


  for stage_m, stage_f in zip(mother, father):
    new_stage=[]
    
    for block_m, block_f in zip(stage_m, stage_f):
      if(np.random.rand()<0.5):
        new_stage.append(block_m)
      else:
        new_stage.append(block_f)
  
    genetic_code.append(new_stage)
  return check_out_channels((stem, genetic_code))

# Class that defines the single exemplar architecture
class Exemplar:
  def __init__(self, in_channels, classes, code):
    self.code=code
    self.net=Net(in_channels, classes, code)
    
    self.logsynflow=None
    self.naswot=None
    self.params=None
    self.flops=None
    self.rank=-1

  def set_all(self, inputs, targets, max_params, max_flops):    
    if self.flops is None:
      flop_analysis=FlopCountAnalysis(self.net, inputs)
      flop_analysis.tracer_warnings('none')
      flops=flop_analysis.total()
      #print(f"flops: {flops}/{max_flops} ({flops/max_flops*100}%)")
      self.flops=flops

    if self.params is None:
      params=parameter_count(model=self.net)[""]
      #print(f"params: {params}/{max_params} ({params/max_params*100}%)")
      self.params=params
    
    if self.logsynflow is None:
      logsynflow=compute_synflow_per_weight(self.net, inputs.device, inputs, targets)
      self.logsynflow=logsynflow

    if self.naswot is None:
      naswot=compute_naswot_score(self.net, inputs.device, inputs, targets)
      self.naswot=naswot

    

# Function that prunes the exemplars that do not satisfy the constraints
def pruneConstraint(exemplars,max_flops,max_params):
  return [e for e in exemplars if (e.flops<=max_flops and e.params<=max_params)]

# Function that select the top K exemplars based on the overall score
def return_top_k(exemplars, K=2):
  values_dict = {}
  values_dict['logsynflow'] = np.array([exemplar.logsynflow for exemplar in exemplars])
  values_dict['naswot'] = np.array([exemplar.naswot for exemplar in exemplars])
  scores = np.zeros(len(exemplars))
  values_dict['logsynflow'] = values_dict['logsynflow'] / (np.max(np.abs(values_dict['logsynflow'])) + 1e-9)
  values_dict['naswot'] = values_dict['naswot'] / (np.max(np.abs(values_dict['naswot'])) + 1e-9)
  scores += values_dict['logsynflow']
  scores += values_dict['naswot']
  for idx, (exemplar, rank) in enumerate(zip(exemplars, scores)):
    exemplar.rank = rank

  exemplars.sort(key=lambda x: -x.rank)
  return exemplars[:K]
  
# Function that executes the hole algorithm
def ProboNAS(N, n, trainloader, max_time, max_params, max_flops, params):
  in_channels=3
  classes=2
  inputs,targets=next(iter(trainloader))

  start=time.time()
  
  exemplars=[]
  print("First Generation creation")

  while (len(exemplars)<N):
    arc_list=randomGen()
    exemplar=Exemplar(in_channels, classes, arc_list)
    exemplar.set_all(inputs, targets, max_params, max_flops)
    exemplars.append(exemplar)
    exemplars=pruneConstraint(exemplars,max_flops,max_params)
  
    
  total_time = 0      
  step = 0

  while total_time <= max_time:
    # Extraction of n random exemplars
    randomlist = random.sample(range(0, len(exemplars)), n)
    exem_group=[exemplars[i] for i in randomlist]
    top2=return_top_k(exem_group)

    # Mutation of the two best exemplars
    for exem in top2:
      #print(exem.code)
      code_mutated=mutation(exem.code, params)
      exem_mutated=Exemplar(in_channels, classes, code_mutated)
      exem_mutated.set_all(inputs, targets, max_params, max_flops)
      exemplars.append(exem_mutated)

    code_crossed=cross(top2[0].code, top2[1].code)
    exem_crossed=Exemplar(in_channels, classes, code_crossed)
    exem_crossed.set_all(inputs, targets, max_params, max_flops)
    exemplars.append(exem_crossed)

    exemplars=pruneConstraint(exemplars,max_flops,max_params)
    
    # Pruning of the worst
    if(len(exemplars)>N):     
      exemplars=return_top_k(exemplars, K=N)   
    

    total_time = (time.time() - start) / 60
    print('time.time: ', time.time())
    print('start time: ', start)
    print('total time: ', total_time)
    step += 1
  
  
  best_exemplar=return_top_k(exemplars, K=1)
  return best_exemplar[0].code


