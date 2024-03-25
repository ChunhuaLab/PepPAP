import numpy as np
import pickle 
import math
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d,self).__init__()
    def forward(self,x):
        output, _ = torch.max(x,1)
        return output

class CNN(nn.Module):
    def __init__(self,in_dim,c_dim,kernel_size):
        super(CNN,self).__init__()
        padding_size = (kernel_size - 1) // 2
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels= c_dim, kernel_size=kernel_size,padding=padding_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=c_dim, out_channels= c_dim*2, kernel_size=kernel_size,padding=padding_size),
            nn.ReLU(),
            nn.Conv1d(in_channels=c_dim*2, out_channels= c_dim*3, kernel_size=kernel_size,padding=padding_size),
            nn.ReLU(),
            #GlobalMaxPool1d() # 192
            )
    def forward(self,x):
        x = self.convs(x)
        return x

class MultiHeadAttention(nn.Module):
    
    def __init__(self, input_dim, d_k, d_v, dropout=0.1):
        n_head=2
        super().__init__()
        
        self.n_head = n_head
        self.input_dim = input_dim
        self.d_k = d_k
        self.d_v = d_v

        self.W_Q = nn.Linear(input_dim, n_head*d_k)
        self.W_K = nn.Linear(input_dim, n_head*d_k)
        self.W_V = nn.Linear(input_dim, n_head*d_v)
        self.W_O = nn.Linear(n_head*d_v, input_dim)

        self.layer_norm = nn.LayerNorm(input_dim)
        
        self.dropout = nn.Dropout(dropout)
    


    def forward(self, q, k, v):
        
        batch, len_q, _ = q.size()
        batch, len_k, _ = k.size()
        batch, len_v, _ = v.size()
        
        Q = self.W_Q(q).view([batch, len_q, self.n_head, self.d_k])
        K = self.W_K(k).view([batch, len_k, self.n_head, self.d_k])
        V = self.W_V(v).view([batch, len_v, self.n_head, self.d_v])
            
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2).transpose(2, 3)
        V = V.transpose(1, 2)

        attention = torch.matmul(Q, K)
       
        attention = attention /np.sqrt(self.d_k)
        attention = F.softmax(attention, dim=-1)
        output = torch.matmul(attention, V)
        output = output.transpose(1, 2).reshape([batch, len_q, self.d_v*self.n_head])
        output = self.W_O(output)
        #output = self.dropout(output)        
        #output = self.layer_norm(output + q)

        return output, attention


class PpIP(nn.Module):
    def __init__(self):
        super(PpIP,self).__init__()
        #self.config = config
        n=512
        self.embed_pep=nn.Embedding(21,128)
        #self.seq_enc=nn.Embedding(30,64)
        self.fc_bert = nn.Linear(1024,128) # padding_idx=0, vocab_size = 65/25, embedding_size=128
        self.fc_intrinsic = nn.Linear(3,128)
        self.fc_phyche = nn.Linear(9,128)
        self.fc_stapot = nn.Linear(3,128)
        self.fc_spot = nn.Linear(21,128)        
        
        self.pep_convs = CNN(n,64,6)
        self.prot_convs = CNN(n,64,8)
        #self.ffn_seq = FFN(64, 64)
        self.global_max_pooling = GlobalMaxPool1d()
        #self.FNN = DNN(config.in_dim,config.d_dim1,config.d_dim2,config.dropout)
        
        self.FNN = nn.Sequential(
            nn.Linear(640,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,512)
        )
        
        #self.sequence_attention_layer = MultiHeadAttentionSequence(4, 64,
        #                                                        64, 128, dropout=0.1)
                
        #self.att = Self_Attention(384,128,128)
        self.mul_att=MultiHeadAttention(128,128,128)
        
        #c_dim
        self.output = nn.Linear(512,1)




    #@torchsnooper.snoop()
    def forward(self, X_pep_intrinsic,X_prot_intrinsic,X_pep_phy,X_prot_phy,peptide_emb,protein_emb, X_pep_stapot,X_prot_stapot):
        #print('X_bert_prot',X_prot_bert.shape)
        #print('intrinsic',X_pep_intrinsic.shape,X_prot_intrinsic.shape)
        #print('phy',X_pep_phy.shape,X_prot_phy.shape)
        

        
        pep_seq_emb = self.embed_pep(peptide_emb.long())#.type(torch.LongTensor))
        pro_seq_emb = self.embed_pep(protein_emb.long())
        
        pep_intrinsic_X = self.fc_intrinsic(X_pep_intrinsic)#type(torch.LongTensor))
        prot_intrinsic_X = self.fc_intrinsic(X_prot_intrinsic)
        
        pep_phyche_X = self.fc_phyche(X_pep_phy)
        prot_phyche_X = self.fc_phyche(X_prot_phy)

        pep_stapot_X = self.fc_stapot(X_pep_stapot)
        prot_stapot_X = self.fc_stapot(X_prot_stapot)
        
        #pep_spot_X = self.fc_spot(X_pep_spot)
        #prot_spot_X = self.fc_spot(X_prot_spot)
        
        

              
        encode_peptide = torch.cat([pep_intrinsic_X, pep_phyche_X,pep_seq_emb,pep_stapot_X],dim=-1)
        encode_protein = torch.cat([prot_intrinsic_X, prot_phyche_X,pro_seq_emb,prot_stapot_X],dim=-1)

        encode_peptide_reshape = encode_peptide.permute(0,2,1)
        encode_protein_reshape = encode_peptide.permute(0,2,1)

        #print('encode_peptide...',encode_peptide.shape,encode_protein.shape)
        # torch.Size([128, 50, 384]) torch.Size([128, 685, 384])
        #print()
        encode_peptide_ori = self.pep_convs(encode_peptide_reshape)
        encode_peptide_ori = encode_peptide_ori.permute(0,2,1)
        encode_peptide_global = self.global_max_pooling(encode_peptide_ori)

        encode_protein_ori = self.prot_convs(encode_protein_reshape)
        encode_protein_ori = encode_protein_ori.permute(0,2,1)
        encode_protein_global = self.global_max_pooling(encode_protein_ori)

        #print('global',encode_peptide_global.shape,encode_protein_global.shape)#torch.Size([20, 192]) torch.Size([20, 192])


        
        # self-attention
       
               
       
        pep_emb = self.embed_pep(peptide_emb.long())#.type(torch.LongTensor))
        pro_emb = self.embed_pep(protein_emb.long()) 
                      
        peptide_out,peptide_att = self.mul_att(pep_emb,pep_emb,pep_emb)
        peptide_out = self.global_max_pooling(peptide_out)
        #print('atten',peptide_out.shape,peptide_att.shape)#atten torch.Size([20, 384]) torch.Size([20, 6, 35, 35])
       
        protein_out,protein_att = self.mul_att(pro_emb,pro_emb,pro_emb)
        protein_out = self.global_max_pooling(protein_out)


        encode_interaction = torch.cat([encode_peptide_global,encode_protein_global,peptide_out,protein_out],axis=-1)
        #print('encode_interaction',encode_interaction.shape)#[20, 115])

        
        encode_interaction = self.FNN(encode_interaction)
        
        predictions =self.output(encode_interaction)
        #print(1)
        return predictions.squeeze(dim=1)
