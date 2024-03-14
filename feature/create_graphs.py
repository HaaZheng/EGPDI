import pickle
import dgl
import torch
from torch_geometric.utils import dense_to_sparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_coor_train(dis_path,query_ids):
    dis_load = open(dis_path, 'rb')
    dis_residue = pickle.load(dis_load)
    dis_residue['4ne1_p'] = dis_residue.pop('4ne1_pp')

    query_ids = query_ids[:573]
    coors = []
    for i in query_ids:
        coor = dis_residue[i]
        coors.append(coor)
    return coors

def get_coor_test(dis_path,query_ids):
    dis_load = open(dis_path, 'rb')
    dis_residue = pickle.load(dis_load)
    query_ids = query_ids[573:]
    coors = []
    for i in query_ids:
        coor = dis_residue[i]
        coors.append(coor)
    return coors

def create_graph(src,dis,device):

    G = dgl.graph((src,dis)).to(device)    # 用dgl.graph版本更快  旧版本使用dgl.DGLGraph

    return G.to(device)

def get_adj(pro_ids):

    adjs = []
    #  create adj_SC_17 based on native structure
    save_files_dir = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/intermediate_files/'
    fpath = save_files_dir +'Train_Test129/'
    adj_type = 'adj_SC_17'
    for i in pro_ids:
        file = fpath + adj_type + '/{}.pkl'.format(i)
        adj_load = open(file, 'rb')
        adj = pickle.load(adj_load)

        adjs.append(adj)

    return adjs

def get_adj_predicted(pro_ids):

    adjs = []
    #  create adj_SC_17 based on predicted structure
    save_files_dir = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/intermediate_files/'
    # 129

    fpath = save_files_dir +'Train_Test129/'
    adj_type = 'adj_SC_17_predicted'
    for i in pro_ids:
        file = fpath + adj_type + '/{}.pkl'.format(i)
        adj_load = open(file, 'rb')
        adj = pickle.load(adj_load)

        adjs.append(adj)

    return adjs