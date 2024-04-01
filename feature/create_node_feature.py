import pickle
import numpy as np
import torch

train_list=[]
seqanno= {}
Query_ids=[]
query_seqs=[]
query_annos=[]


def one_hot_encode(sequence):
    # create one-hot
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    encoded_sequence = np.zeros((len(sequence), len(amino_acids)), dtype=np.float32)
    for i, aa in enumerate(sequence):
        encoded_sequence[i, aa_to_int[aa]] = 1
    return encoded_sequence


def create_features(query_ids,all_702_path,train_path,test_path,pkl_path,esm2_33_path,esm2_5120_path,ProtTrans_path,msa_256_path):
    '''
        load node features :
        1) Esm2-t48(5120dim),Esm2-t36(33dim),ProtTrans(1024dim),
        2) Residual_feats(HMM-20dim,PSSM-30dim,AF-7dim,DSSP-14dim)
    '''

    # load protein_id,protein,seq,protein_label
    with open(train_path,'r') as f:
        train_text = f.readlines()
        for i in range(0, len(train_text), 4):
            query_id = train_text[i].strip()[1:]
            # if query_id[-1].islower():
            #     query_id += query_id[-1]
            query_seq = train_text[i + 1].strip()
            query_anno = train_text[i + 2].strip()
            train_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}  
            Query_ids.append(query_id)
            query_seqs.append(query_seq)

    with open(test_path, 'r') as f:
        train_text = f.readlines()
        for i in range(0, len(train_text), 3):
            query_id = train_text[i].strip()[1:]
            # if query_id[-1].islower():
            #     query_id += query_id[-1]
            #     print(query_id,'-'*1000)
            query_seq = train_text[i + 1].strip()
            query_anno = train_text[i + 2].strip()
            train_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}
            Query_ids.append(query_id)
            query_seqs.append(query_seq)

    # load one-hot
    query_seqs_702 = []
    with open(all_702_path, 'r') as f1:
        text_702 = f1.readlines()
        for i in range(1, len(text_702), 3):      
            query_seq_702 = text_702[i].strip()
            query_seqs_702.append(query_seq_702)
    encoded_proteins = [one_hot_encode(sequence) for sequence in query_seqs_702]

    # load Residual_feats-71dim
    PDNA_residue_load=open(pkl_path,'rb')
    PDNA_residue=pickle.load(PDNA_residue_load)
    PDNA_residue['4ne1_p'] = PDNA_residue.pop('4ne1_pp')

    
    # load esm2-t36 embeddings-33dim
    ESM2_33 = []
    paths = []
    for i in query_ids:
        file_paths = esm2_33_path + '{}'.format(i) + '.npy'
        paths.append(file_paths)
    for file_path in paths:
        ESM2_33_embedding = np.load(file_path)
        ESM2_33.append(ESM2_33_embedding)

    # load esm2-t48 embeddings-5120dim
    ESM2_5120 = []
    paths_5120 = []
    for i in query_ids:
        # file_paths = esm2_5120_path + '{}'.format(i) + '.rep_5120.npy'  
        file_paths = esm2_5120_path + '{}'.format(i) + '.npy'
        paths_5120.append(file_paths)
    for file_path in paths_5120:
        # print(file_path)
        ESM2_5120_embedding = np.load(file_path,allow_pickle=True)
        ESM2_5120.append(ESM2_5120_embedding)

    # load MSA embeddings-256dim
    MSA_256 = []
    paths_256 = []
    for i in query_ids:
        # file_paths = esm2_5120_path + '{}'.format(i) + '.rep_5120.npy'  
        file_paths = msa_256_path + '{}'.format(i) + 'msa_first_row.npy'
        paths_256.append(file_paths)
    for file_path in paths_256:
        # print(file_path)
        msa_256_embedding = np.load(file_path,allow_pickle=True)
        MSA_256.append(msa_256_embedding)

    # load ProtTrans embeddings-1024dim
    ProTrans_1024=[]
    paths_1024 = []
    for i in query_ids:
        file_paths = ProtTrans_path + '{}'.format(i) + '.pt'
        paths_1024.append(file_paths)
    for file_path in paths_1024:
        ProTrans_1024_embedding = torch.load(file_path)
        ProTrans_1024.append(ProTrans_1024_embedding)

    # load residue features-71dim and labels
    data = {}
    for i in query_ids:
        data[i] = []
        residues = PDNA_residue[i]
        labels = seqanno[i]['anno']
        data[i].append({'features': residues, 'label': labels})


    feature1=[]
    feature2=[]
    feature3=[]
    feature4 = []
    feature5 = []
    feature6 = []

    protein_labels=[]

    for i in query_ids:

        residues=data[i]
        feature1.append(residues[0]['features'])
        protein_labels.append((residues[0]['label']))

    for j in range(len(query_ids)):
        feature2.append(encoded_proteins[j])   # 20 dim one-hot
        feature3.append(ESM2_5120[j])    # 5120 dim bert_5120
        feature4.append(ProTrans_1024[j])  # 1024 dim protrans
        feature5.append(ESM2_33[j])
        feature6.append(MSA_256[j])

    node_features={}
    for i in range(len(query_ids)):
        # node_features[query_ids[i]]={'seq': i+1,'residue_fea': feature1[i],'esm2_33':feature5[i],'prottrans_1024':feature4[i],'one-hot':feature2[i],'label':protein_labels[i]}
        node_features[query_ids[i]]={'seq': i+1,'residue_fea': feature1[i],'esm2_5120':feature3[i],'esm2_33':feature5[i],'prottrans_1024':feature4[i],'msa_256':feature6[i],'one-hot':feature2[i],'label':protein_labels[i]}

    return node_features


def create_dataset(query_ids,train_path, test_path,all_702_path, pkl_path,esm2_33_path,esm2_5120_path,ProtTrans_path,msa_256_path,residue,one_hot,esm2_33,esm_5120,prottrans,msa):
    '''
    :param query_ids: all protein ids
    :param train_path: training set file path
    :param test_path: test_129 set file path
    :param all_702_path: train_573 and test_129 file path
    :param pkl_path: residue features path
    :param esm2_33_path: esm2-t36 embeddings path
    :param esm2_5120_path: esm2-t48 embeddings path
    :param ProtTrans_path: ProtTrans embeddings path
    :param residue: add residue features or not
    :param one_hot: add one-hot features or not
    :param esm2_33: add esm2-t36 features or not
    :param esm_5120: add esm2-t48 features or not
    :param prottrans: add ProtTrans features or not
    :return: X and y, involving training and validation set
    '''

    X=[]
    y=[]
    features={}

    # all 702 protein information
    node_features = create_features(query_ids,all_702_path,train_path,test_path,pkl_path,esm2_33_path,esm2_5120_path,ProtTrans_path,msa_256_path)

    for i in query_ids:
        protein = node_features[i]

        mat1 = (protein['residue_fea'])
        mat2 = (protein['one-hot'])
        mat3 = (protein['esm2_33'])
        mat4 = (protein['esm2_5120'])
        mat5 = (protein['prottrans_1024'])
        mat6 = (protein['msa_256'])

        mat4 = torch.Tensor(mat4)
        mat4 = torch.squeeze(mat4)

        mat5 = torch.Tensor(mat5)
        mat5 = torch.squeeze(mat5)

        # different feature combinations
        # handy drafted features and protein language model embeddings
        if residue == True and one_hot == True and esm2_33 == True and esm_5120 == True and prottrans == True and msa == True :   # all embeddings
            features[i] = np.hstack((mat1,mat2,mat3,mat4,mat5,mat6))
        # handy drafted features
        elif residue == True and one_hot == True and esm2_33 == False and esm_5120 == False and prottrans == False and msa == False :    # only hand drafted features protein language model embeddings
            features[i] = np.hstack((mat1,mat2))
        # protein language model embeddings
        elif residue == False and one_hot == False and esm2_33 == True and esm_5120 == True and prottrans == True and msa == True :    # only protein language model embeddings
            features[i] = np.hstack((mat3,mat4,mat5,mat6))
        elif residue == True and one_hot == True and esm2_33 == True and esm_5120 == False and prottrans == False and msa == False:
            features[i] = np.hstack((mat1,mat2,mat3))
        elif residue == True and one_hot == True and esm2_33 == False and esm_5120 == False and prottrans == True and msa == False:
            features[i] = np.hstack((mat1,mat2,mat5))
        elif residue == True and one_hot == True and esm2_33 == False and esm_5120 == True and prottrans == False and msa == False:
            features[i] = np.hstack((mat1,mat2,mat4))
        elif residue == True and one_hot == True and esm2_33 == False and esm_5120 == False and prottrans == False and msa == True:
            features[i] = np.hstack((mat1,mat2,mat6))
        elif residue == True and one_hot == True and esm2_33 == False and esm_5120 == False and prottrans == True and msa == True:
            features[i] = np.hstack((mat1,mat2,mat5,mat6))
        elif residue == True and one_hot == True and esm2_33 == False and esm_5120 == True and prottrans == True and msa == True:
            features[i] = np.hstack((mat1,mat2,mat4,mat5,mat6))

        labels = protein['label']
        y.append(labels)

    for key in query_ids:
        X.append(features[key])


    return X,y







