import numpy as np
import pickle
import os,sys




amino_acid_set = { "A": 1, "C": 2, "Y": 3, "E": 4, "D": 5, "G": 6, 
        "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, 
        "V": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18, 
        "W": 19, "T": 20} # consider non-standard residues


amino_acid_num = 20
def label_sequence(line, pad_prot_len, res_ind):
  X = np.zeros(pad_prot_len)

  for i, res in enumerate(line[:pad_prot_len]):
    X[i] = res_ind[res]

  return X
 
pad_pep_len = 35 
pad_prot_len = 300    

proseq=sys.argv[1]
pepseq=sys.argv[2]


      
feature_pep = label_sequence(pepseq, pad_pep_len, amino_acid_set)
#data_pep[pep]=feature_pep
feature_pro = label_sequence(proseq, pad_prot_len, amino_acid_set)


np.save('./example/peptide_enc.npy',feature_pep)
np.save('./example/protein_enc.npy',feature_pro)
