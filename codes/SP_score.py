import os,sys
import pickle
import numpy as np


sys.path.append('./utils') 
import pading_features as pad


dssp_codes = {
    'T': 'X', 'G': 'X', ' ': 'X',
    'I': 'Y', 'S': 'Y',
    'H': 'Z', 'E': 'Z', 'B': 'Z'
}



residues=['A','V','I','L','M','F','W','Y','S','T','N','Q','R','H','K','D','E','C','G','P']

    
def feature(typ,seq):
    
    with open(f'./codes/{typ}_20_dict','rb') as f:
        data_dict = pickle.load(f)   
              
        
     

    P_feature=np.zeros([1, 3])

    for residue in seq:
        if residue in residues:               
           P_residue=data_dict[residue]
           #print(P_residue)
           temp=[ sum(P_residue)/len(P_residue),max(P_residue),min(P_residue)]
           temp=np.array(temp).reshape(1,3)
           P_feature=np.concatenate((P_feature,temp),axis=0)
        else:   
           P_residue=data_dict['X']
           temp=[ sum(P_residue)/len(P_residue),max(P_residue),min(P_residue)]
           P_feature=np.concatenate((P_feature,temp),axis=0)                 

    P_feature=np.delete(P_feature,0,axis=0)   
    P_feature=np.asarray(P_feature, dtype = float)
       
    F=pad.pading(typ,P_feature)
    print(F.shape)
   
    np.save('./example/{}_SP_score.npy'.format(typ),F)

proseq=sys.argv[1]
pepseq=sys.argv[2]

feature('protein',proseq)
feature('peptide',pepseq)