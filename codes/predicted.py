import numpy as np
import pickle 
import math
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
sys.path.append('./units') 
from modules import *

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True




peptide_emb=np.load('example/peptide_enc.npy')
peptide_emb=np.array(peptide_emb).astype(int).reshape(1, -1)
protein_emb=np.load('example/protein_enc.npy')
protein_emb=np.array(protein_emb).astype(int).reshape(1, -1)

X_pep_intrinsic=np.load('example/peptide_intrinsic.npy')
X_pep_intrinsic = np.array(X_pep_intrinsic).astype(np.float32)
X_pep_intrinsic = np.expand_dims(X_pep_intrinsic, axis=0)
X_prot_intrinsic=np.load('example/protein_intrinsic.npy')
X_prot_intrinsic = np.array(X_prot_intrinsic).astype(np.float32)
X_prot_intrinsic = np.expand_dims(X_prot_intrinsic, axis=0)

X_pep_stapot=np.load('example/peptide_SP_score.npy')
X_pep_stapot = np.array(X_pep_stapot).astype(np.float32)
X_pep_stapot = np.expand_dims(X_pep_stapot, axis=0)
X_prot_stapot=np.load('example/protein_SP_score.npy')
X_prot_stapot = np.array(X_prot_stapot).astype(np.float32)
X_prot_stapot = np.expand_dims(X_prot_stapot, axis=0)

X_pep_phy=np.load('example/peptide_PhyChe.npy')
X_pep_phy = np.array(X_pep_phy).astype(np.float32)
X_pep_phy = np.expand_dims(X_pep_phy, axis=0)
X_prot_phy=np.load('example/protein_PhyChe.npy')
X_prot_phy = np.array(X_prot_phy).astype(np.float32)
X_prot_phy = np.expand_dims(X_prot_phy, axis=0)

#Y = np.array(Y).astype(np.float32)



#print('X_emb',peptide_emb.shape,protein_emb.shape)
#print('X_intrinsic',X_pep_intrinsic.shape,X_prot_intrinsic.shape)
#print('X_phy',X_pep_phy.shape,X_prot_phy.shape)
#print('X_stapot',X_pep_stapot.shape,X_prot_stapot.shape)





def load_checkpoint(filepath):
    M = torch.load(filepath)
    model = M['model']
    model.load_state_dict(M['model_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad=False
    model.eval()
    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )



    
       
preds =0
    
for n in range(1,6):

    model= load_checkpoint(f'./models/model{n}.pkl') # for inference, can refer to these few lines.

    model = model.to(device)
    model.eval()



    pred=model((torch.from_numpy(X_pep_intrinsic)).to(device),(torch.from_numpy(X_prot_intrinsic)).to(device),
    (torch.from_numpy(X_pep_phy)).to(device),(torch.from_numpy(X_prot_phy)).to(device),
    (torch.from_numpy(peptide_emb)).to(device),(torch.from_numpy(protein_emb)).to(device),
    (torch.from_numpy(X_pep_stapot)).to(device),(torch.from_numpy(X_prot_stapot)).to(device),
    )
    

    pred = pred.detach().cpu().numpy()
    preds +=pred


pred = preds[0]/5


pred_pK = np.round(pred,3)
pred_dg=np.round((pred_pK*(-0.59)*math.log(10)),3)


out=open('./results/predected_result.txt','a')

out.write(f"The predicted pK value is:{pred_pK} \n")
out.write(f"The predicted dg (kcal/mol) value is:{pred_dg} \n")
out.close()

print(f"The predicted pK value is:{pred_pK}")
print(f"The predicted dg (kcal/mol) value is:{pred_dg}")

#np.savetxt('pred_results.txt',pred_dg)





