#!/bin/bash

#PDB ID 1G6R (protein chain C and peptide chain Q)

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


