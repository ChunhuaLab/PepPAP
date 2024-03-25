import os 
import sys 
import struct
import numpy as np
import pandas as pd
import pickle

sys.path.append('./utils') 
import pading_features as pad

   
    
def feature(typ,seq):

    with open('./codes/AA_dict','rb') as f:
        AA_dict = pickle.load(f)   
          

    Residues=list(AA_dict.keys())
   
    PhyChe_feature_dict = {}
    


            
    AA_feature=np.zeros([1, 9])

    for residue in seq:
        if residue in Residues:               
           AA_residue=AA_dict[residue]
           AA_feature=np.concatenate((AA_feature,AA_residue),axis=0)
        else:   
           AA_residue=AA_dict['X']
           AA_feature=np.concatenate((AA_feature,AA_residue),axis=0)                 

    AA_feature=np.delete(AA_feature,0,axis=0)   
    PhyChe_feature=np.asarray(AA_feature, dtype = float)

    
    F=pad.pading(typ, PhyChe_feature)
    print(F.shape)
    np.save('./example/{}_PhyChe.npy'.format(typ),F)
    


proseq=sys.argv[1]
pepseq=sys.argv[2]

feature('protein',proseq)
feature('peptide',pepseq)