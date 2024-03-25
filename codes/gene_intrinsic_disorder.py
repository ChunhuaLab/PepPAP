import sys
import os,pickle
import numpy as np

sys.path.append('./utils') 
import pading_features as pad


def gen_file_results(typ,seq):


     os.system('python ./softwares/iupred2a/iupred2a.py {} {} long'.format(typ,seq))
     os.system('python ./softwares/iupred2a/iupred2a.py -a {} {} short'.format(typ,seq))
    #os.system('mv ./out/* ./out_{}'.format(typ))
    
    

def extract_intrinsic_disorder(nameid,seq,ind):

    disorder_filename = nameid+'_'+ind+'.result'
    
    raw_fasta_list = [nameid,seq]

    
    fasta_id_list = [nameid]
    fasta_sequence_list = [seq]
    fasta_seq_len_list = [len(seq)]
    #print(len(fasta_id_list),len(fasta_sequence_list),len(fasta_seq_len_list))
    
    fasta_dict={}
    fasta_dict[nameid]=(seq,len(seq))

    # load protein intrinsic disorder result
    raw_result_list = []
    with open('./softwares/iupred2a/out/{}'.format(disorder_filename),'r') as f:
            for line in f.readlines():
                line_list = line.strip()
                if (len(line_list)>0 and line_list[0]!='#'):
                    raw_result_list.append(line_list)
 

    intrinsic_id_list = [x for x in raw_result_list if x[0]=='>']
    intrinsic_score_list = [x.split('\t') for x in raw_result_list if x[0]!='>']

    start_idx = 0
    raw_score_dict = {}
    

    prot_id = nameid
    seq_len = len(seq)
    end_idx = start_idx + seq_len
    individual_score_list = intrinsic_score_list[start_idx:end_idx]
    individual_score_list=[x[2:] for x in individual_score_list]
    individual_score_array = np.array(individual_score_list,dtype='float')
        
    raw_score_dict[prot_id] = individual_score_array
    start_idx = end_idx
    #print(len(fasta_dict.keys()),len(raw_score_dict.keys()))
    
    
    return fasta_dict, raw_score_dict

def feature(Type,seq):
    
     
        # long & short
    fasta_dict_long, raw_score_dict_long = extract_intrinsic_disorder(Type,seq,'long') # the input fasta file used in IUPred2A
    fasta_dict_short, raw_score_dict_short = extract_intrinsic_disorder(Type,seq,'short')

    #print(fasta_dict_short)
    #print(raw_score_dict_short)

    Intrinsic_score_long = {}
    for key in fasta_dict_long.keys():    
        sequence = fasta_dict_long[key][0]
        seq_len = fasta_dict_long[key][1]
        Intrinsic = raw_score_dict_long[key]
        if Intrinsic.shape[0]!= seq_len:
            #print(nameid,Intrinsic.shape[0],seq_len)
            print('Error!')
        Intrinsic_score_long[sequence]= Intrinsic
        
    #print(Intrinsic_score_long)   
     
    Intrinsic_score_short = {}
    for key in fasta_dict_short.keys():
        sequence = fasta_dict_short[key][0]
        seq_len = fasta_dict_short[key][1]
        Intrinsic = raw_score_dict_short[key]
        if Intrinsic.shape[0]!= seq_len:
            print('Error!')
        Intrinsic_score_short[sequence] = Intrinsic
       


    
    for seq in Intrinsic_score_short.keys():
        long_Intrinsic = Intrinsic_score_long[seq][:,0]
        short_Intrinsic = Intrinsic_score_short[seq]
        concat_Intrinsic = np.column_stack((long_Intrinsic,short_Intrinsic))
        Intrinsic_score = np.column_stack((long_Intrinsic,short_Intrinsic))

    #print(Type,Intrinsic_score)
    feature=pad.pading(Type,Intrinsic_score)
    np.save('./example/{}_intrinsic.npy'.format(Type),feature)
    #with open('./example/{}_intrinsic_dict'.format(Type),'wb') as f: # 'output_intrisic_dict' is the name of the output dict you like
    #    pickle.dump(Intrinsic_score,f)
  

    #return Intrinsic_score



proseq=sys.argv[1]
pepseq=sys.argv[2]

gen_file_results('protein',proseq)
gen_file_results('peptide',pepseq)   

Intrinsic_pro=feature('protein',proseq)  
Intrinsic_pep=feature('peptide',pepseq)  


    
