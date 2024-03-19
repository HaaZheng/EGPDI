# 2023.12.13
'''
main python file
1) load dataset
2) call model function
3) call training function
4) call evaluation function
5) call test function  # next step
'''
# done
# without ESM2-t48,average mcc=0.462

import time
from torch.utils.data import Dataset
from sklearn.metrics import matthews_corrcoef,  precision_score, recall_score, f1_score,roc_auc_score, average_precision_score,confusion_matrix
from torch_geometric.utils import dense_to_sparse
from sklearn.model_selection import KFold
import dgl
import pandas as pd
from torch.autograd import Variable
from torch.utils import data

from model import *
from feature.create_node_feature import create_dataset
from feature.create_graphs import get_coor_train,get_adj
from feature.create_edge import create_dis_matrix


import warnings
warnings.filterwarnings("ignore")
seed_value = 1995
th=17


features = []
labels = []

class CustomDataset(Dataset):

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


root_dir = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/raw_files/Train_Test129/'
train_path= root_dir + 'DNA-573_Train.txt'
test_path= root_dir + 'DNA-129_Test.txt'
all_702_path = root_dir +  'DNA-702.txt'
pkl_path= root_dir + 'PDNA_residue_feas_PHSA.pkl'
esm2_5120_path= root_dir + 'ESM2-t48/'
# esm2_5120_path= root_dir + 'ESM2_t48_Equi/'
esm2_33_path = root_dir + 'ESM2-t36/'
dis_path= root_dir + 'PDNA_psepos_SC.pkl'
ProtTrans_path = root_dir +  'ProtTrans/'
mas_path = root_dir + 'MSA/'


query_ids = []
with open(train_path, 'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 4):
        query_id = train_text[i].strip()[1:]
        # if query_id[-1].islower():
        #     query_id += query_id[-1]
        query_ids.append(query_id)
with open(test_path, 'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        query_id = train_text[i].strip()[1:]
        # if query_id[-1].islower():
        #     query_id += query_id[-1]
        #     # print(query_id, '-' * 1000)
        query_ids.append(query_id)

print(query_ids)

X,y = create_dataset(query_ids,train_path, test_path,all_702_path, pkl_path,esm2_33_path,esm2_5120_path,ProtTrans_path,mas_path,residue=True,one_hot=True,esm2_33=False,esm_5120=False,prottrans=False,msa=False)

distance_matrixs=create_dis_matrix(dis_path,query_ids)

X_train = X[:573]
X_test = X[573:]

y_train = y[:573]
y_test = y[573:]


INPUT_DIM = 71+20
HIDDEN_DIM = 512
LAYER = 2
DROPOUT = 0.3
ALPHA = 0.1
LAMBDA = 1.0
VARIANT = True
LEARNING_RATE = 0.0001
NUMBER_EPOCHS = 35

# 构建五倍交叉验证适用的数据

IDs = query_ids[:573]

sequences = []
labels = []
with open(all_702_path,'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        seq = train_text[i+1].strip()
        label = train_text[i+2].strip()
        sequences.append(seq)
        labels.append(label)

sequences = sequences[:573]


labels = y_train
features = X_train

coors = get_coor_train(dis_path, query_ids)
adjs = get_adj(IDs)

graphs = []
for adj in adjs:
    edge_index, _ = dense_to_sparse(adj)
    G = dgl.graph((edge_index[0], edge_index[1])).to(device)
    graphs.append(G)

train_dic = {"ID": IDs, "sequence": sequences, "label": labels,'features':features,'coors':coors,'adj':adjs,'graph':graphs}
dataframe = pd.DataFrame(train_dic)


class dataSet(data.Dataset):

    def __init__(self,dataframe,adjs):

        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.features = dataframe['features'].values
        self.coors = dataframe['coors'].values
        self.graphs =  dataframe['graph'].values
        self.adj = dataframe['adj'].values

    def __getitem__(self,index):

        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = self.labels[index]
        node_features = self.features[index]
        coors = self.coors
        coor = coors[index]
        graphs = self.graphs
        graph = graphs[index]
        adj = self.adj[index]

        return sequence_name,sequence,label,node_features,graph,adj,coor


    def __len__(self):

        return len(self.labels)


def graph_collate(samples):

    _,_,label_batch, node_features_batch, graph_batch,adj_batch,coors_batch = map(list, zip(*samples))

    graph_batch = dgl.batch(graph_batch)

    return label_batch, node_features_batch, graph_batch,adj_batch,coors_batch



from sklearn.model_selection import GridSearchCV
def grid_search(all_dataframe):

    X_train = all_dataframe['features'].values
    y_train = all_dataframe['label'].values

    model = MainModel().to(device)
    param_grid = {
        'dropout': [0.0001, 0.00005, 0.00001],

    }

    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    print(grid_search.best_params_)


aver_epoch = grid_search(dataframe)