import pickle
import pandas as pd
import numpy as np
import os
import sys
import shutil
from tqdm import tqdm, trange
import random
import torch
from torch_geometric.data import InMemoryDataset, Data
import prettytable as pt
import math
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--ligand", dest="ligand", default='DNA',
                        help="A ligand type. It can be chosen from DNA,RNA,CA,MG,MN,ATP,HEME.")
    parser.add_argument("--psepos", dest="psepos", default='SC',
                        help="Pseudo position of residues. SC, CA, C stand for centroid of side chain, alpha-C atom and centroid of residue, respectively.")
    parser.add_argument("--features", dest="features", default='PSSM,HMM,SS,AF',
                        help="Feature groups. Multiple features should be separated by commas. You can combine features from PSSM, HMM, SS(secondary structure) and AF(atom features).")
    parser.add_argument("--context_radius", default=11, dest="context_radius", type=int,
                        help="Radius of structure context.")
    parser.add_argument("--trans_anno", dest="trans_anno", type=bool, default=True,
                        help="Transfer binding annotations for DNA-(RNA-)binding protein training data sets or not.")
    parser.add_argument("--tvseed", dest='tvseed', type=int, default=1995,
                        help='The random seed used to separate the validation set from training set.')
    return parser.parse_args()


def checkargs(args):
    if args.ligand not in ['DNA', 'RNA', 'CA', 'MN', 'MG', 'ATP', 'HEME']:
        print('ERROR: ligand "{}" is not supported by GraphBind!'.format(args.ligand))
        raise ValueError
    if args.psepos not in ['SC', 'CA', 'C']:
        print('ERROR: pseudo position of a residue "{}" is not supported by GraphBind!'.format(args.psepos))
        raise ValueError
    features = args.features.strip().split(',')
    for feature in features:
        if feature not in ['PSSM', 'HMM', 'SS', 'AF']:
            print('ERROR: feature "{}" is not supported by GraphBind!'.format(feature))
            raise ValueError
    if args.context_radius <= 0:
        print('ERROR: radius of structure context should be positive!')
        raise ValueError

    return


class NeighResidue3DPoint(InMemoryDataset):
    def __init__(self, root, dataset, transform=None, pre_transform=None):
        super(NeighResidue3DPoint, self).__init__(root, transform, pre_transform)

        if dataset == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
        elif dataset == 'valid':
            self.data, self.slices = torch.load(self.processed_paths[1])
        elif dataset == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])

    @property
    def raw_file_names(self):
        splits = ['train', 'valid', 'test']
        return ['{}_data.pkl'.format(s) for s in splits]

    @property
    def processed_file_names(self):
        return ['train.pt', 'valid.pt', 'test.pt']

    def _download(self):
        pass

    def process(self):

        seq_data_dict = {}
        for s, dataset in enumerate(['train', 'valid', 'test']):
            data_list = []
            with open(self.raw_dir + '/{}_data.pkl'.format(dataset), 'rb') as f:
                [data_dict, seqlist] = pickle.load(f)
            for seq in tqdm(seqlist):
                seq_data_list = []
                seq_data = data_dict[seq]
                for res_data in seq_data:
                    node_feas = res_data['node_feas']
                    node_feas = torch.tensor(node_feas, dtype=torch.float32)
                    pos = torch.tensor(res_data['pos'], dtype=torch.float32)
                    label = torch.tensor([res_data['label']], dtype=torch.float32)
                    data = Data(x=node_feas, pos=pos, y=label)
                    seq_data_list.append(data)
                data_list.extend(seq_data_list)
                seq_data_dict[seq] = seq_data_list
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[s])
        torch.save(seq_data_dict, root_dir + '/processed/seq_data_dict.pt')


def Create_NeighResidue3DPoint(psepos, dist, feature_dir, raw_dir, seqanno, feature_combine, train_list, valid_list,
                               test_list):
    with open(feature_dir + '/' + ligand + '_psepos_{}.pkl'.format(psepos), 'rb') as f:
        residue_psepos = pickle.load(f)
    with open(feature_dir + '/' + ligand + '_residue_feas_{}.pkl'.format(feature_combine), 'rb') as f:
        residue_feas = pickle.load(f)

    for s, (dataset, seqlist) in enumerate(zip(['train', 'valid', 'test'],
                                               [train_list, valid_list, test_list])):
        data_dict = {}
        for seq in tqdm(seqlist):
            seq_data = []
            feas = residue_feas[seq]
            pos = residue_psepos[seq]
            label = np.array(list(map(int, list(seqanno[seq]['anno']))))

            for i in range(len(label)):
                res_psepos = pos[i]
                res_dist = np.sqrt(np.sum((pos - res_psepos) ** 2, axis=1))
                neigh_index = np.where(res_dist < dist)[0]
                res_atom_id = np.arange(len(neigh_index))
                id_dict = dict(list(zip(neigh_index, res_atom_id)))
                res_pos = pos[neigh_index] - res_psepos
                res_feas = feas[neigh_index]

                res_label = label[i]
                res_data = {'node_feas': res_feas.astype('float32'),
                            'pos': res_pos.astype('float32'),
                            'label': res_label.astype('float32'),
                            'neigh_index': neigh_index.astype('int32')}
                seq_data.append(res_data)
            data_dict[seq] = seq_data
        with open(raw_dir + '/{}_data.pkl'.format(dataset), 'wb') as f:
            pickle.dump([data_dict, seqlist], f)

    return


def def_atom_features():
    A = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 3, 0]}
    V = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 3, 0],
         'CG2': [0, 3, 0]}
    F = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 1, 1]}
    P = {'N': [0, 0, 1], 'CA': [0, 1, 1], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 1], 'CG': [0, 2, 1],
         'CD': [0, 2, 1]}
    L = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 1, 0],
         'CD1': [0, 3, 0], 'CD2': [0, 3, 0]}
    I = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'CG1': [0, 2, 0],
         'CG2': [0, 3, 0], 'CD1': [0, 3, 0]}
    R = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 2, 0], 'CD': [0, 2, 0], 'NE': [0, 1, 0], 'CZ': [1, 0, 0], 'NH1': [0, 2, 0], 'NH2': [0, 2, 0]}
    D = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [-1, 0, 0],
         'OD1': [-1, 0, 0], 'OD2': [-1, 0, 0]}
    E = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [-1, 0, 0], 'OE1': [-1, 0, 0], 'OE2': [-1, 0, 0]}
    S = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'OG': [0, 1, 0]}
    T = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 1, 0], 'OG1': [0, 1, 0],
         'CG2': [0, 3, 0]}
    C = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'SG': [-1, 1, 0]}
    N = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 0, 0],
         'OD1': [0, 0, 0], 'ND2': [0, 2, 0]}
    Q = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 0, 0], 'OE1': [0, 0, 0], 'NE2': [0, 2, 0]}
    H = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'ND1': [-1, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'NE2': [-1, 1, 1]}
    K = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'CD': [0, 2, 0], 'CE': [0, 2, 0], 'NZ': [0, 3, 1]}
    Y = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 1, 1], 'CE1': [0, 1, 1], 'CE2': [0, 1, 1], 'CZ': [0, 0, 1],
         'OH': [-1, 1, 0]}
    M = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0], 'CG': [0, 2, 0],
         'SD': [0, 0, 0], 'CE': [0, 3, 0]}
    W = {'N': [0, 1, 0], 'CA': [0, 1, 0], 'C': [0, 0, 0], 'O': [0, 0, 0], 'CB': [0, 2, 0],
         'CG': [0, 0, 1], 'CD1': [0, 1, 1], 'CD2': [0, 0, 1], 'NE1': [0, 1, 1], 'CE2': [0, 0, 1], 'CE3': [0, 1, 1],
         'CZ2': [0, 1, 1], 'CZ3': [0, 1, 1], 'CH2': [0, 1, 1]}
    G = {'N': [0, 1, 0], 'CA': [0, 2, 0], 'C': [0, 0, 0], 'O': [0, 0, 0]}

    atom_features = {'A': A, 'V': V, 'F': F, 'P': P, 'L': L, 'I': I, 'R': R, 'D': D, 'E': E, 'S': S,
                     'T': T, 'C': C, 'N': N, 'Q': Q, 'H': H, 'K': K, 'Y': Y, 'M': M, 'W': W, 'G': G}
    for atom_fea in atom_features.values():
        for i in atom_fea.keys():
            i_fea = atom_fea[i]
            atom_fea[i] = [i_fea[0] / 2 + 0.5, i_fea[1] / 3, i_fea[2]]

    return atom_features


def get_pdb_DF(file_path):
    atom_fea_dict = def_atom_features()
    res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
                'TRP': 'W', 'CYS': 'C',
                'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E',
                'LYS': 'K', 'ARG': 'R','UNK': 'U'}
    atom_count = -1
    res_count = -1
    pdb_file = open(file_path, 'r')
    pdb_res = pd.DataFrame(columns=['ID', 'atom', 'res', 'res_id', 'xyz', 'B_factor'])
    res_id_list = []
    before_res_pdb_id = None
    Relative_atomic_mass = {'H': 1, 'C': 12, 'O': 16, 'N': 14, 'S': 32, 'FE': 56, 'P': 31, 'BR': 80, 'F': 19, 'CO': 59,
                            'V': 51,
                            'I': 127, 'CL': 35.5, 'CA': 40, 'B': 10.8, 'ZN': 65.5, 'MG': 24.3, 'NA': 23, 'HG': 200.6,
                            'MN': 55,
                            'K': 39.1, 'AP': 31, 'AC': 227, 'AL': 27, 'W': 183.9, 'SE': 79, 'NI': 58.7}

    while True:
        line = pdb_file.readline()
        if line.startswith('ATOM'):
            atom_type = line[76:78].strip()
            if atom_type not in Relative_atomic_mass.keys():
                continue
            atom_count += 1
            res_pdb_id = int(line[22:26])

            # if res_pdb_id != before_res_pdb_id:
            #     res_count += 1

            before_res_pdb_id = res_pdb_id
            if line[12:16].strip() not in ['N', 'CA', 'C', 'O', 'H']:
                is_sidechain = 1
            else:
                is_sidechain = 0
            res = res_dict[line[17:20]]
            atom = line[12:16].strip()
            try:
                atom_fea = atom_fea_dict[res][atom]
            except KeyError:
                atom_fea = [0.5, 0.5, 0.5]
            tmps = pd.Series(
                {'ID': atom_count, 'atom': line[12:16].strip(), 'atom_type': atom_type, 'res': res,
                 'res_id': int(line[22:26]),
                 'xyz': np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])]),
                 'occupancy': float(line[54:60]),
                 'B_factor': float(line[60:66]), 'mass': Relative_atomic_mass[atom_type], 'is_sidechain': is_sidechain,
                 'charge': atom_fea[0], 'num_H': atom_fea[1], 'ring': atom_fea[2]})
            if len(res_id_list) == 0:
                res_id_list.append(int(line[22:26]))
            elif res_id_list[-1] != int(line[22:26]):
                res_id_list.append(int(line[22:26]))
            pdb_res = pdb_res.append(tmps, ignore_index=True)

        if line.startswith('TER'):
            break

    return pdb_res, res_id_list


def cal_PDBDF(seqlist, PDB_chain_dir, PDB_DF_dir):
    if not os.path.exists(PDB_DF_dir):
        os.mkdir(PDB_DF_dir)

    for seq_id in tqdm(seqlist):
        # print(seq_id)
        file_path = PDB_chain_dir + '/{}.pdb'.format(seq_id)
        with open(file_path, 'r') as f:
            text = f.readlines()
        if len(text) == 1:
            print('ERROR: PDB {} is empty.'.format(seq_id))
        if  os.path.exists(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id)):
            try:
                pdb_DF, res_id_list = get_pdb_DF(file_path)
                with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'wb') as f:
                    pickle.dump({'pdb_DF': pdb_DF, 'res_id_list': res_id_list}, f)
            except KeyError:
                print('ERROR: UNK in ', seq_id)
                # raise KeyError

    return


def cal_Psepos(seqlist, PDB_DF_dir, Dataset_dir, psepos, ligand, seqanno):
    seq_CA_pos = {}
    seq_centroid = {}
    seq_sidechain_centroid = {}

    for seq_id in tqdm(seqlist):
        print(seq_id)
        with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'rb') as f:
            tmp = pickle.load(f)
        pdb_res_i, res_id_list = tmp['pdb_DF'], tmp['res_id_list']

        res_CA_pos = []
        res_centroid = []
        res_sidechain_centroid = []
        res_types = []
        for res_id in res_id_list:
            res_type = pdb_res_i[pdb_res_i['res_id'] == res_id]['res'].values[0]
            res_types.append(res_type)

            res_atom_df = pdb_res_i[pdb_res_i['res_id'] == res_id]
            xyz = np.array(res_atom_df['xyz'].tolist())
            masses = np.array(res_atom_df['mass'].tolist()).reshape(-1, 1)
            centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
            res_sidechain_atom_df = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['is_sidechain'] == 1)]

            try:
                CA = pdb_res_i[(pdb_res_i['res_id'] == res_id) & (pdb_res_i['atom'] == 'CA')]['xyz'].values[0]
            except IndexError:
                #     print('IndexError: no CA in seq:{} res_id:{}'.format(seq_id,res_id))
                CA = centroid

            res_CA_pos.append(CA)
            res_centroid.append(centroid)

            if len(res_sidechain_atom_df) == 0:
                res_sidechain_centroid.append(centroid)
            else:
                xyz = np.array(res_sidechain_atom_df['xyz'].tolist())
                masses = np.array(res_sidechain_atom_df['mass'].tolist()).reshape(-1, 1)
                sidechain_centroid = np.sum(masses * xyz, axis=0) / np.sum(masses)
                res_sidechain_centroid.append(sidechain_centroid)

        if ''.join(res_types) != seqanno[seq_id]['seq']:
            print(seq_id)
            print(''.join(res_types))
            print(seqanno[seq_id]['seq'])
            return
        res_CA_pos = np.array(res_CA_pos)
        res_centroid = np.array(res_centroid)
        res_sidechain_centroid = np.array(res_sidechain_centroid)
        seq_CA_pos[seq_id] = res_CA_pos
        seq_centroid[seq_id] = res_centroid
        seq_sidechain_centroid[seq_id] = res_sidechain_centroid

    if psepos == 'CA':
        with open(Dataset_dir + '/' + ligand + '_psepos_' + psepos + '.pkl', 'wb') as f:
            pickle.dump(seq_CA_pos, f)
    elif psepos == 'C':
        with open(Dataset_dir + '/' + ligand + '_psepos_' + psepos + '.pkl', 'wb') as f:
            pickle.dump(seq_centroid, f)
    elif psepos == 'SC':
        with open(Dataset_dir + '/' + ligand + '_psepos_' + psepos + '.pkl', 'wb') as f:
            pickle.dump(seq_sidechain_centroid, f)

    return


def cal_PSSM(ligand, seq_list, pssm_dir, feature_dir):
    nor_pssm_dict = {}
    for seqid in seq_list:
        file = seqid + '.pssm'
        with open(pssm_dir + '/' + file, 'r') as fin:
            fin_data = fin.readlines()
            pssm_begin_line = 3
            pssm_end_line = 0
            for i in range(1, len(fin_data)):
                if fin_data[i] == '\n':
                    pssm_end_line = i
                    break
            feature = np.zeros([(pssm_end_line - pssm_begin_line), 20])
            axis_x = 0
            for i in range(pssm_begin_line, pssm_end_line):
                raw_pssm = fin_data[i].split()[2:22]
                axis_y = 0
                for j in raw_pssm:
                    feature[axis_x][axis_y] = (1 / (1 + math.exp(-float(j))))
                    axis_y += 1
                axis_x += 1
            nor_pssm_dict[file.split('.')[0]] = feature
    with open(feature_dir + '/{}_PSSM.pkl'.format(ligand), 'wb') as f:
        pickle.dump(nor_pssm_dict, f)
    return


def cal_HMM(ligand, seq_list, hmm_dir, feature_dir):
    hmm_dict = {}
    for seqid in seq_list:
        file = seqid + '.hhm'
        with open(hmm_dir + '/' + file, 'r') as fin:
            fin_data = fin.readlines()
            hhm_begin_line = 0
            hhm_end_line = 0
            for i in range(len(fin_data)):
                if '#' in fin_data[i]:
                    hhm_begin_line = i + 5
                elif '//' in fin_data[i]:
                    hhm_end_line = i
            feature = np.zeros([int((hhm_end_line - hhm_begin_line) / 3), 30])
            axis_x = 0
            for i in range(hhm_begin_line, hhm_end_line, 3):
                line1 = fin_data[i].split()[2:-1]
                line2 = fin_data[i + 1].split()
                axis_y = 0
                for j in line1:
                    if j == '*':
                        feature[axis_x][axis_y] = 9999 / 10000.0
                    else:
                        feature[axis_x][axis_y] = float(j) / 10000.0
                    axis_y += 1
                for j in line2:
                    if j == '*':
                        feature[axis_x][axis_y] = 9999 / 10000.0
                    else:
                        feature[axis_x][axis_y] = float(j) / 10000.0
                    axis_y += 1
                axis_x += 1
            feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
            hmm_dict[file.split('.')[0]] = feature
    with open(feature_dir + '/{}_HMM.pkl'.format(ligand), 'wb') as f:
        pickle.dump(hmm_dict, f)
    return


# 定义了两个字典：maxASA和map_ss_8。
# maxASA：存储了20种氨基酸的最大相对表面积（Relative Accessible Surface Area, ASA）值。这些值用于计算二级结构中疏水性残基的相对可及性。
# map_ss_8：将8种常见的二级结构符号映射为8维的二进制编码，用于表示二级结构类型
# 构建包含各特征的numpy数组feature，并将其保存在seq_feature字典中，以氨基酸残基的序号res_id为键，对应的特征数组为值。
# 前8维特征表示二级结构类型，使用map_ss_8进行二进制编码。
# 第9维特征表示疏水性残基的相对可及性，通过将相对表面积的比值归一化到[0, 1]范围内。
# 第10至13维特征表示二面角信息，将其归一化到[0, 1]范围内。
def cal_DSSP(ligand, seq_list, dssp_dir, feature_dir):
    maxASA = {'G': 188, 'A': 198, 'V': 220, 'I': 233, 'L': 304, 'F': 272, 'P': 203, 'M': 262, 'W': 317, 'C': 201,
              'S': 234, 'T': 215, 'N': 254, 'Q': 259, 'Y': 304, 'H': 258, 'D': 236, 'E': 262, 'K': 317, 'R': 319}
    map_ss_8 = {' ': [1, 0, 0, 0, 0, 0, 0, 0], 'S': [0, 1, 0, 0, 0, 0, 0, 0], 'T': [0, 0, 1, 0, 0, 0, 0, 0],
                'H': [0, 0, 0, 1, 0, 0, 0, 0],
                'G': [0, 0, 0, 0, 1, 0, 0, 0], 'I': [0, 0, 0, 0, 0, 1, 0, 0], 'E': [0, 0, 0, 0, 0, 0, 1, 0],
                'B': [0, 0, 0, 0, 0, 0, 0, 1]}
    dssp_dict = {}
    for seqid in seq_list:
        file = seqid + '.dssp'
        with open(dssp_dir + '/' + file, 'r') as fin:
            fin_data = fin.readlines()
        seq_feature = {}
        for i in range(25, len(fin_data)):
            line = fin_data[i]
            if line[13] not in maxASA.keys() or line[9] == ' ':
                continue
            res_id = float(line[5:10])
            feature = np.zeros([14])
            feature[:8] = map_ss_8[line[16]]
            feature[8] = min(float(line[35:38]) / maxASA[line[13]], 1)
            feature[9] = (float(line[85:91]) + 1) / 2
            feature[10] = min(1, float(line[91:97]) / 180)
            feature[11] = min(1, (float(line[97:103]) + 180) / 360)
            feature[12] = min(1, (float(line[103:109]) + 180) / 360)
            feature[13] = min(1, (float(line[109:115]) + 180) / 360)
            seq_feature[res_id] = feature.reshape((1, -1))
        dssp_dict[file.split('.')[0]] = seq_feature
    with open(feature_dir + '/{}_SS.pkl'.format(ligand), 'wb') as f:
        pickle.dump(dssp_dict, f)
    return


def PDBResidueFeature(seqlist, PDB_DF_dir, feature_dir, ligand, residue_feature_list, feature_combine, atomfea):
    for fea in residue_feature_list:
        with open(feature_dir + '/' + ligand + '_{}.pkl'.format(fea), 'rb') as f:
            locals()['residue_fea_dict_' + fea] = pickle.load(f)

    atom_vander_dict = {'C': 1.7, 'O': 1.52, 'N': 1.55, 'S': 1.85, 'H': 1.2, 'D': 1.2, 'SE': 1.9, 'P': 1.8, 'FE': 2.23,
                        'BR': 1.95,
                        'F': 1.47, 'CO': 2.23, 'V': 2.29, 'I': 1.98, 'CL': 1.75, 'CA': 2.81, 'B': 2.13, 'ZN': 2.29,
                        'MG': 1.73, 'NA': 2.27,
                        'HG': 1.7, 'MN': 2.24, 'K': 2.75, 'AC': 3.08, 'AL': 2.51, 'W': 2.39, 'NI': 2.22}
    for key in atom_vander_dict.keys():
        atom_vander_dict[key] = (atom_vander_dict[key] - 1.52) / (1.85 - 1.52)

    residue_feas_dict = {}
    for seq_id in tqdm(seqlist):
        # print(seq_id)
        with open(PDB_DF_dir + '/{}.csv.pkl'.format(seq_id), 'rb') as f:
            tmp = pickle.load(f)

        pdb_res_i, res_id_list = tmp['pdb_DF'], tmp['res_id_list']
        pdb_res_i = pdb_res_i[pdb_res_i['atom_type'] != 'H']
        mass = np.array(pdb_res_i['mass'].tolist()).reshape(-1, 1)
        mass = mass / 32

        B_factor = np.array(pdb_res_i['B_factor'].tolist()).reshape(-1, 1)
        if (max(B_factor) - min(B_factor)) == 0:
            B_factor = np.zeros(B_factor.shape) + 0.5
        else:
            B_factor = (B_factor - min(B_factor)) / (max(B_factor) - min(B_factor))
        is_sidechain = np.array(pdb_res_i['is_sidechain'].tolist()).reshape(-1, 1)
        occupancy = np.array(pdb_res_i['occupancy'].tolist()).reshape(-1, 1)
        charge = np.array(pdb_res_i['charge'].tolist()).reshape(-1, 1)
        num_H = np.array(pdb_res_i['num_H'].tolist()).reshape(-1, 1)
        ring = np.array(pdb_res_i['ring'].tolist()).reshape(-1, 1)

        atom_type = pdb_res_i['atom_type'].tolist()
        atom_vander = np.zeros((len(atom_type), 1))
        for i, type in enumerate(atom_type):
            try:
                atom_vander[i] = atom_vander_dict[type]
            except:
                atom_vander[i] = atom_vander_dict['C']

        atom_feas = [mass, B_factor, is_sidechain, charge, num_H, ring, atom_vander]  # 原子特征的具体特征种类
        atom_feas = np.concatenate(atom_feas, axis=1)
        residue_feas = []
        for fea in residue_feature_list:
            fea_i = locals()['residue_fea_dict_' + fea][seq_id]
            if isinstance(fea_i, np.ndarray):
                residue_feas.append(fea_i)
            elif isinstance(fea_i, dict):
                fea_ii = []
                for res_id_i in res_id_list:
                    if res_id_i in fea_i.keys():
                        fea_ii.append(fea_i[res_id_i])
                    else:
                        fea_ii.append(np.zeros(list(fea_i.values())[0].shape))
                fea_ii = np.concatenate(fea_ii, axis=0)
                residue_feas.append(fea_ii)
        try:
            residue_feas = np.concatenate(residue_feas, axis=1)
        except ValueError:
            print('ERROR: Feature dimensions of {} are inconsistent!'.format(seq_id))
            raise ValueError
        if residue_feas.shape[0] != len(res_id_list):
            print(
                'ERROR: For {}, the number of residues with features is not consistent with the number of residues in the query!'.format(
                    seq_id))
            raise IndexError

        if atomfea:
            res_atom_feas = []
            atom_begin = 0
            for i, res_id in enumerate(res_id_list):
                res_atom_df = pdb_res_i[pdb_res_i['res_id'] == res_id]
                atom_num = len(res_atom_df)
                res_atom_feas_i = atom_feas[atom_begin:atom_begin + atom_num]
                res_atom_feas_i = np.average(res_atom_feas_i, axis=0).reshape(1, -1)
                res_atom_feas.append(res_atom_feas_i)
                atom_begin += atom_num
            res_atom_feas = np.concatenate(res_atom_feas, axis=0)
            residue_feas = np.concatenate((res_atom_feas, residue_feas), axis=1)

        residue_feas_dict[seq_id] = residue_feas

    with open(feature_dir + '/' + ligand + '_residue_feas_' + feature_combine + '.pkl', 'wb') as f:
        pickle.dump(residue_feas_dict, f)

    return


def tv_split(train_list, seed):
    random.seed(seed)
    random.shuffle(train_list)
    valid_list = train_list[:int(len(train_list) * 0.2)]
    train_list = train_list[int(len(train_list) * 0.2):]
    return train_list, valid_list


def StatisticsSampleNum( test_list, seqanno):
    def sub(seqlist, seqanno):
        pos_num_all = 0
        res_num_all = 0
        for seqid in seqlist:
            anno = list(map(int, list(seqanno[seqid]['anno'])))
            pos_num = sum(anno)
            res_num = len(anno)
            pos_num_all += pos_num
            res_num_all += res_num
        neg_num_all = res_num_all - pos_num_all
        pnratio = pos_num_all / float(neg_num_all)
        return len(seqlist), res_num_all, pos_num_all, neg_num_all, pnratio

    tb = pt.PrettyTable()
    tb.field_names = ['Dataset', 'NumSeq', 'NumRes', 'NumPos', 'NumNeg', 'PNratio']
    tb.float_format = '0.3'

    seq_num, res_num, pos_num, neg_num, pnratio = sub(test_list, seqanno)
    tb.add_row(['test', seq_num, res_num, pos_num, neg_num, pnratio])
    print(tb)
    return


if __name__ == '__main__':

    args = parse_args()
    checkargs(args)

    ligand = 'P' + args.ligand if args.ligand != 'HEME' else 'PHEM'
    psepos = args.psepos
    trans_anno = args.trans_anno
    dist = args.context_radius
    feature_list = []
    feature_combine = ''
    if 'PSSM' in args.features:
        feature_list.append('PSSM')
        feature_combine += 'P'
    if 'HMM' in args.features:
        feature_list.append('HMM')
        feature_combine += 'H'
    if 'SS' in args.features:
        feature_list.append('SS')
        feature_combine += 'S'
    if 'AF' in args.features:
        feature_list.append('AF')
        feature_combine += 'A'



    testset_dict = {'PDNA': 'DNA_Test_181.txt',

                    }

    Dataset_dir = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/raw_files/Test_181'
    PDB_chain_dir = Dataset_dir + '/PDB'

    testset_anno = Dataset_dir + '/{}'.format(testset_dict[ligand])

    seqanno = {}
    train_list = []
    test_list = []

    if ligand in ['PDNA', 'PRNA']:
        with open(testset_anno, 'r') as f:
            test_text = f.readlines()
        for i in range(0, len(test_text), 3):
            query_id = test_text[i].strip()[1:]
            if query_id[-1].islower():
                query_id += query_id[-1]
            query_seq = test_text[i + 1].strip()
            query_anno = test_text[i + 2].strip()
            test_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}
    else:
        with open(testset_anno, 'r') as f:
            test_text = f.readlines()
        for i in range(0, len(test_text), 3):
            query_id = test_text[i].strip()[1:]
            query_seq = test_text[i + 1].strip()
            query_anno = test_text[i + 2].strip()
            test_list.append(query_id)
            seqanno[query_id] = {'seq': query_seq, 'anno': query_anno}

    # train_list, valid_list = tv_split(train_list, args.tvseed)
    StatisticsSampleNum(test_list, seqanno)

    PDB_DF_dir = Dataset_dir + '/PDB_DF'
    seqlist = test_list

    print('1.Extract the PDB information.')
    cal_PDBDF(seqlist, PDB_chain_dir, PDB_DF_dir)
    print('2.calculate the pseudo positions.')
    cal_Psepos(seqlist, PDB_DF_dir, Dataset_dir, psepos, ligand, seqanno)
    print('3.calculate the residue features.')
    if 'AF' in feature_list:
        atomfea = True
        feature_list.remove('AF')
    else:
        atomfea = False

    cal_PSSM(ligand, seqlist, Dataset_dir + '/PSSM', Dataset_dir)
    cal_HMM(ligand, seqlist, Dataset_dir + '/HMM', Dataset_dir)
    cal_DSSP(ligand, seqlist, Dataset_dir + '/SS', Dataset_dir)

    PDBResidueFeature(seqlist, PDB_DF_dir, Dataset_dir, ligand, feature_list, feature_combine, atomfea)

    root_dir = Dataset_dir + '/' + ligand + '_{}_dist{}_{}'.format(psepos, dist, feature_combine)
    raw_dir = root_dir + '/raw'
    if os.path.exists(raw_dir):
        shutil.rmtree(root_dir)
    os.makedirs(raw_dir)
    print('4.Calculate the neighborhood of residues. Save to {}.'.format(root_dir))
    Create_NeighResidue3DPoint(psepos, dist, Dataset_dir, raw_dir, seqanno, feature_combine, test_list)
    _ = NeighResidue3DPoint(root=root_dir, dataset='test')
