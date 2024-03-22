# EGPDI
Identifying Protein-DNA binding sites based on multi-view graph embedding fusion

Data preparation
1. Generate multiple protein node features (HMM, PSSM, AF, One hot encoding, SS) by running the data_io.py file in the script folder.
2. Generate embedding of ESM2 and ProtTrans protein language models, and the generated files can be found here: https://drive.google.com/drive/my-drive
3. Generate the adjacency matrix of the protein by running the create_adj_predict.py file in the script folder.

Training
1. Run train_val_bestAUPR_predicted.py for model training

Test
1. Run test_129_final.py for model testing
2. Run test_181_final.py for independent test set testing.
   
Pre training model acquisition can be found in the link: https://drive.google.com/drive/my-drive
