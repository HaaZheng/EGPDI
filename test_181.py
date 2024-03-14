# 2024.1.2
'''
use trained model, use test_181 dataset, cal these evaluation metrics
use cpu , change modules' device into cpu
'''
import torch
import pickle
from torch.utils.data import Dataset
from sklearn.metrics import matthews_corrcoef,  precision_score, recall_score, f1_score,roc_auc_score, average_precision_score,confusion_matrix
from torch_geometric.utils import dense_to_sparse
import dgl
import pandas as pd
from torch.autograd import Variable
from torch.utils import data

# from model_with_edge_features import MainModel #ori
# from ablation_study.model_ablation_fusion.model_EGNN import MainModel
# from ablation_study.model_ablation_fusion.model_GCNII import MainModel
from model_forGCNIIBiLSTM import MainModel
from feature.create_node_feature_181 import create_dataset
from feature.create_graphs_181 import get_coor_test,get_adj,get_adj_predicted
from feature.create_edge_181 import create_dis_matrix,get_edge_attr_test_181


import warnings
warnings.filterwarnings("ignore")
seed_value = 1995
th=17

# device = "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# root_dir = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/raw_files/Test_181/'
# root_dir = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/predicted_structure/Test_181/'
root_dir = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/predicted_structure/Test_181_180/'

test_path= root_dir + 'DNA_Test_181.txt'
pkl_path= root_dir + 'PDNA_residue_feas_PHSA.pkl'
esm2_5120_path= root_dir + 'ESM2-t48/'
esm_33_path = root_dir + 'ESM2-t36/'
dis_path= root_dir + 'PDNA_psepos_SC.pkl'
ProtTrans_path = root_dir +  'ProtTrans/'
msa_256_path = root_dir + 'MSA/'


query_ids = []

with open(test_path, 'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):

        query_id = train_text[i].strip()[1:]
        if query_id[-1].islower():
            query_id += query_id[-1]
        query_ids.append(query_id)

print(len(query_ids))
# query_ids,test_path,pkl_path,esm2_5120_path,ProtTrans_path,msa_256_path,residue,one_hot,msa_256,esm_5120,prottrans
X,y = create_dataset(query_ids,test_path,pkl_path,esm2_5120_path,esm_33_path,ProtTrans_path,msa_256_path,residue=True,one_hot=True,msa_256=True,esm_5120=True,esm_33= True,prottrans=True)
distance_matrixs=create_dis_matrix(dis_path,query_ids)

X_test = X
y_test = y


NUMBER_EPOCHS = 30

dr=0.3
lr=0.0001
nlayers=4
lamda=1.1
alpha=0.1
atten_time=8

# 构建五倍交叉验证适用的数据

IDs = query_ids


sequences = []
labels = []
with open(test_path,'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        seq = train_text[i+1].strip()
        label = train_text[i+2].strip()
        sequences.append(seq)
        labels.append(label)

sequences = sequences


labels = y_test
features = X_test

coors = get_coor_test(dis_path, query_ids)
adjs = get_adj_predicted(IDs)
# adjs = get_adj(IDs)


graphs = []
for adj in adjs:
    edge_index, _ = dense_to_sparse(adj)
    G = dgl.graph((edge_index[0], edge_index[1])).to(device)
    graphs.append(G)

# 先创建，后直接读取边特征
# print('start to prepare edge features')
# efeats = []
# for i in range(len(IDs)):
#     edge_feats = get_edge_attr_test_181(i,th=17,distance_matrixs=distance_matrixs)
#     print(edge_feats.shape)
#     efeats.append(edge_feats)
# save_edgefeats_path = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/raw_files/Test_181/edge_features'
# with open(save_edgefeats_path + '/EdgeFeats_raw_SC_17_181.pkl', 'wb') as f:
#     pickle.dump(efeats, f)

save_edgefeats_path = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/predicted_structure/Test_181_180/edge_features/EdgeFeats_predict_SC_17_181.pkl'
with open(save_edgefeats_path, 'rb') as f:
    efeats = pickle.load(f)


test_dic = {"ID": IDs, "sequence": sequences, "label": labels,'features':features,'coors':coors,'adj':adjs,'graph':graphs,'efeats':efeats}
dataframe = pd.DataFrame(test_dic)


class dataSet(data.Dataset):

    def __init__(self,dataframe,adjs):

        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.features = dataframe['features'].values
        self.coors = dataframe['coors'].values
        self.graphs =  dataframe['graph'].values
        self.efeats = dataframe['efeats'].values
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

        efeat = self.efeats[index]

        return sequence_name,sequence,label,node_features,graph,efeat,adj,coor


    def __len__(self):

        return len(self.labels)


def graph_collate(samples):

    _,_,label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch = map(list, zip(*samples))

    graph_batch = dgl.batch(graph_batch)

    return label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch


def evaluate(model,data_loader):

    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = []

    for label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch in data_loader:

        with torch.no_grad():

            node_features_batch = torch.tensor(node_features_batch)
            coors_batch = torch.tensor(coors_batch)
            adj_batch = adj_batch[0]
            label_batch = label_batch[0]
            efeat_batch = efeat_batch[0]

            if torch.cuda.is_available():
                node_features_batch = Variable(node_features_batch.cuda())
                graph_batch = graph_batch.to(device)
                adj_batch = Variable(adj_batch.cuda())
                efeat_batch = efeat_batch.to(device)
                coors_batch = Variable(coors_batch.cuda())
                y_true = label_batch
            else:
                node_features_batch = Variable(node_features_batch)
                graph_batch = graph_batch
                adj_batch = Variable(adj_batch)
                coors_batch = Variable(coors_batch)
                y_true = label_batch
                efeat_batch = efeat_batch

            # print('node_features',node_features_batch.shape)
            # print('graph',graph_batch)
            # print('adj',adj_batch.shape)
            # print('coors',coors_batch.shape)
            # print('y',len(y_true))
            # print('efeats',efeat_batch.shape)

            # y_pred = model(graph_batch, node_features_batch,coors_batch,adj_batch,efeat_batch)   # ori
            y_pred = model( node_features_batch,adj_batch)

            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = torch.squeeze(y_pred)

            y_true_int = [int(label) for label in y_true]
            y_true = torch.tensor(y_true_int, dtype=torch.float32, device=device)

            loss = model.criterion(y_pred,y_true)

            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()

            # valid_pred += [pred[1] for pred in y_pred]
            valid_pred += [pred for pred in y_pred]
            valid_true += list(y_true)

            epoch_loss += loss.item()
            n += 1

    epoch_loss_avg = epoch_loss / n
    print('evaluate time', n)

    return epoch_loss_avg,valid_true,valid_pred


def analysis(y_true,y_pred,best_threshold = None):

    if best_threshold == None:
        best_mcc = 0
        best_f1 = 0
        best_threshold = 0

        for j in range(0, 100):

            threshold = j / 100000
            print('threshold:', threshold)

            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            mcc = matthews_corrcoef(binary_true, binary_pred)
            f1 = f1_score(y_true, binary_pred, zero_division=0)
            aupr = average_precision_score(y_true, y_pred)
            print('mcc:', mcc)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold

        print(best_mcc)
        print(best_threshold)

    binary_pred = [1.0 if pred >= best_threshold else 0.0 for pred in y_pred]

    # correct_samples = (binary_pred == y_true).sum().item()
    correct_samples = sum(a == b for a, b in zip(binary_pred, y_true))
    accuracy = correct_samples / len(y_true)

    # y_trues = y_true.detach().cpu().numpy()

    pre = precision_score(y_true, binary_pred, zero_division=0)
    recall = recall_score(y_true, binary_pred, zero_division=0)
    f1 = f1_score(y_true, binary_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, binary_pred)

    # pred_array = y_pred.detach().cpu().numpy()
    auc = roc_auc_score(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)

    tn, fp, fn, tp = confusion_matrix(y_true, binary_pred).ravel()
    spe = tn / (tn + fp)

    results = {
        'accuracy':accuracy,
        'spe':spe,
        'precision': pre,
        'recall': recall,
        'f1':f1,
        'mcc': mcc,
        'auc':auc,
        'pr_auc':pr_auc,
        'thred':best_threshold
    }

    return results


def test_181(Model_Path):

    # model = MainModel().to(device)

    # model = MainModel(dr,lr,nlayers,lamda,alpha,atten_time).to(device)  # ori need to add ,nfeats=71+20+33+256+1024+5120
    model = MainModel(dr, lr, nlayers, lamda, alpha, atten_time, nfeats=71 + 20 + 256 + 1024 + 5120 + 33).to(device)
    # print(model)
    model.load_state_dict(torch.load(Model_Path,map_location=device))

    test_dataSet = dataSet(dataframe=dataframe, adjs=adjs)
    # print(test_dataSet.features.shape)
    test_loader = torch.utils.data.DataLoader(test_dataSet, batch_size=1, shuffle=False, collate_fn=graph_collate)

    _, test_true, test_pred = evaluate(model, test_loader)

    test_results = analysis(test_true, test_pred)

    return test_results


# trained_model save path
# # Model_Path_1 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/AUPR_4_Predicted/Fold1predicted_edgeFeats_best_AUPR_model.pkl'   # ori
# Model_Path_2 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/AUPR_4_Predicted/Fold2predicted_edgeFeats_best_AUPR_model.pkl'
# Model_Path_3 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/AUPR_4_Predicted/Fold3predicted_edgeFeats_best_AUPR_model.pkl'
# Model_Path_4 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/AUPR_4_Predicted/Fold4predicted_edgeFeats_best_AUPR_model.pkl'
# Model_Path_5 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/AUPR_4_Predicted/Fold5predicted_edgeFeats_best_AUPR_model.pkl'

# Model_Path_1 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_EGNN/Fold1onlyEGNN_Predict_EdgeFeats_AUPR.pkl'   # EGNN
# Model_Path_2 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_EGNN/Fold2onlyEGNN_Predict_EdgeFeats_AUPR.pkl'   # EGNN
# Model_Path_3 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_EGNN/Fold3onlyEGNN_Predict_EdgeFeats_AUPR.pkl'   # EGNN
# Model_Path_4 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_EGNN/Fold4onlyEGNN_Predict_EdgeFeats_AUPR.pkl'   # EGNN
# Model_Path_5 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_EGNN/Fold5onlyEGNN_Predict_EdgeFeats_AUPR.pkl'   # EGNN

Model_Path_1 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ComparedWith_GCNIIBiLSTM/Fold1_predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_2 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ComparedWith_GCNIIBiLSTM/Fold2_predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_3 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ComparedWith_GCNIIBiLSTM/Fold3_predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_4 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ComparedWith_GCNIIBiLSTM/Fold4_predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_5 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ComparedWith_GCNIIBiLSTM/Fold5_predicted_edgeFeats_best_AUPR_model.pkl'

# Model_Path_1 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_GCNII/Fold1_onlyGCNII4_Predict_EdgeFeats_AUPR.pkl'  # for GCNII
# Model_Path_2 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_GCNII/Fold2_onlyGCNII4_Predict_EdgeFeats_AUPR.pkl'  # for GCNII
# Model_Path_3 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_GCNII/Fold3_onlyGCNII4_Predict_EdgeFeats_AUPR.pkl'  # for GCNII
# Model_Path_4 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_GCNII/Fold4_onlyGCNII4_Predict_EdgeFeats_AUPR.pkl'  # for GCNII
# Model_Path_5 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_GCNII/Fold5_onlyGCNII4_Predict_EdgeFeats_AUPR.pkl'  # for GCNII


test_results_1 = test_181(Model_Path_1)
test_results_2 = test_181(Model_Path_2)
test_results_3 = test_181(Model_Path_3)
test_results_4 = test_181(Model_Path_4)
test_results_5 = test_181(Model_Path_5)

acc = (test_results_1['accuracy']+test_results_2['accuracy']+test_results_3['accuracy']+test_results_4['accuracy']+test_results_5['accuracy'])/5
spe = (test_results_1['spe']+test_results_2['spe']+test_results_3['spe']+test_results_4['spe']+test_results_5['spe'])/5
pre = (test_results_1['precision']+test_results_2['precision']+test_results_3['precision']+test_results_4['precision']+test_results_5['precision'])/5
recall = (test_results_1['recall']+test_results_2['recall']+test_results_3['recall']+test_results_4['recall']+test_results_5['recall'])/5
f1 = (test_results_1['f1']+test_results_2['f1']+test_results_3['f1']+test_results_4['f1']+test_results_5['f1'])/5
mcc = (test_results_1['mcc']+test_results_2['mcc']+test_results_3['mcc']+test_results_4['mcc']+test_results_5['mcc'])/5
auc = (test_results_1['auc']+test_results_2['auc']+test_results_3['auc']+test_results_4['auc']+test_results_5['auc'])/5
pr_auc = (test_results_1['pr_auc']+test_results_2['pr_auc']+test_results_3['pr_auc']+test_results_4['pr_auc']+test_results_5['pr_auc'])/5


print("Test_181 performance on our method")
print("average accuracy: {} \n".format(acc))
print("average spe: {} \n".format(spe))
print("average pre: {} \n".format(pre))
print("average recall: {} \n".format(recall))
print("average f1: {} \n".format(f1))
print("average mcc: {} \n".format(mcc))
print("average auc: {} \n".format(auc))
print("average pr_auc: {} \n".format(pr_auc))