# A deep attention model for wide-genome protein-peptide binding affinity prediction at sequence level.

We develop a model called PepPAP, which is an effective predictor for wide-genome protein-peptide binding affinity prediction at sequence level based on convolutions neural network and multi-head attention.

Authors: Xiaohan Sun, Zhixiang Wu, Jingjie Su, Chunhua Li. 

The performance process includes two steps: installation and prediction.

Here, we take a protein-peptide complex with PDB ID 1G6R for example to show the prediction process, whose binding affinity is 4.27 (pKd/pKi) or ¦¤G = 5.80 kcal/mol. 

In the following descriptions, ¡°proseq¡± and ¡°pepseq¡± denote protein and peptide sequence, respectively.

The protein (chain C) and peptide (chain Q) sequences of 1G6R are the following:

proseq='QSVTQPDARVTVSEGASLQLRCKYSYSATPYLFWYVQYPRQGLQLLLKYYSGDPVVQGVNGFEAEFSKSNSSFHLRKASVHWSDSAVYFCAVSGFASALTFGSGTKVIVLPYIQNPEPAVYALKDPRSQDSTLCLFTDFDSQINVPKTMESGTFITDATVLDMKAMDSKSNGAIAWSNQTSFTCQDIFKETNATYPSSDVPC'

pepseq='SIYRYYGL'

## Step 1 Installation

* Python version: 3.8

  pip install biopython ==1.78; scikit-learn ==1.2.2; pip install torch == 1.7.1; pip install numpy == 1.19.4; pip install scipy == 1.5.4

* Iupred2a
  
  The software is download from https://iupred2a.elte.hu/, it has been given in the folder ¡°softwares¡±.

## Step 2 Prediction

* Place the protein and peptide sequences of 1G6R in their designated positions within the run.sh file. 

* Run the following commands:
  
  proseq='QSVTQPDARVTVSEGASLQLRCKYSYSATPYLFWYVQYPRQGLQLLLKYYSGDPVVQGVNGFEAEFSKSNSSFHLRKASVHWSDSAVYFCAVSGFASALTFGSGTKVIVLPYIQNPEPAVYALKDPRSQDSTLCLFTDFDSQINVPKTMESGTFITDATVLDMKAMDSKSNGAIAWSNQTSFTCQDIFKETNATYPSSDVPC'

  pepseq='SIYRYYGL'

  echo  Begin to extract features from protein and peptide !
  
  cp ./codes/gene_intrinsic_disorder.py ./softwares/iupred2a/
  
  python ./softwares/iupred2a/gene_intrinsic_disorder.py $proseq $pepseq
  
  python ./codes/gen_env_padding_feature.py $proseq $pepseq
  
  python ./codes/Physicochemical_characteristics.py $proseq $pepseq
  
  python ./codes/SP_score.py $proseq $pepseq
  
  echo  Begin to extract features from protein and peptide!
  
  echo The calculation has been completed !
  
  echo Begin to predicte binding affinity value !
  
  python ./codes/predicted.py

Or directly run the following command:

  ./run.sh
 
The finally output is shown in "./results/predected_result.txt".

The predicted (pKd/pKi) value is: 4.204 

The predicted ¦¤G (kcal/mol) value is:-5.711 

## Training PepPAP

Follow the steps in the prediction section to extract features.

After extracting the features, to train PepPAP run the following command. 

  cd PepPAP/utils
  
  python train.py 



## Help

For any questions, please contact us by chunhuali@bjut.edu.cn.
