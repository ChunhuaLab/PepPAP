import numpy as np
import sys
import pickle
import math




def padding_PhyChe_disorder(x,N):
  padding_array = np.zeros([N,x.shape[1]])
  if x.shape[0]>=N: # sequence is longer than N
    padding_array[:N,:x.shape[1]] = x[:N,:]
  else:
    padding_array[:x.shape[0],:x.shape[1]] = x
  return padding_array



    

def pading(typ,feature):


  if typ=='protein':

     padlen = 300 
  else:
  	
     padlen = 35 

  F = padding_PhyChe_disorder(feature, padlen)

  
  return F


