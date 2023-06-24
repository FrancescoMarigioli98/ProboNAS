# The script returns a randomly generated genetic code for the architecture. 


from numpy.random import randint
def randomGen():
  listStruct=[3,3,9,3]
  #listStruct=[2,2,6,2]
  listChannels=[10,20,40,80,160]
  listkernels1=[1,3,5,7]
  listkernels2=[3,5,7]
  listratio=[2,3,4]

  stem_channels=randint(3,listChannels[0]+1)

  stages_list=[]
  for k,j in enumerate(listStruct):
    blocks_list=[]
    a=listChannels[k+1]
    b=listChannels[k]
    for i in range(j):
      genes_list=[]
      x=randint(0,2)
      if x==0:
        genes_list.append('c')
      else:
        genes_list.append('i')
      y=randint(b,a)
      genes_list.append(y)
      b=y
      genes_list.append(randint(0,4))
        
      genes_list.append(listratio[randint(0,len(listratio)-1)])
      blocks_list.append(genes_list)
    stages_list.append(blocks_list)

  return (stem_channels,stages_list)