import torch
from torch.utils.data import Dataset
from sklearn.metrics import matthews_corrcoef,  precision_score, recall_score, f1_score,roc_auc_score, average_precision_score,confusion_matrix
from torch_geometric.utils import dense_to_sparse
import dgl
import pickle
import pandas as pd
from torch.autograd import Variable
from torch.utils import data

# from model import *
# from model_afterGridSearch import *
# from model_withFeatureAnalysis import *
# from model_with_edge_features import MainModel  # 原本
# from ablation_study.model_ablation_fusion.model_EGNN import MainModel # for EGNN
# from ablation_study.model_ablation_fusion.model_GCNII import MainModel # for GCNII
from model_forGCNIIBiLSTM import MainModel
# from ablation_study.model_ablation_fusion.model_EGNN import MainModel  # model ablation, only EGNN
from feature.create_node_feature import create_dataset
from feature.create_graphs import get_coor_test,get_adj_predicted,get_adj
from feature.create_edge import create_dis_matrix,get_edge_attr_test

import warnings
warnings.filterwarnings("ignore")
seed_value = 1995
th=17

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device ='cpu'

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

# 原来的pdb
# root_dir = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/raw_files/Train_Test129/'
# predicted pdb
root_dir = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/predicted_structure/Train_Test129/'

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
        query_ids.append(query_id)
with open(test_path, 'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        query_id = train_text[i].strip()[1:]
        query_ids.append(query_id)


X,y = create_dataset(query_ids,train_path, test_path,all_702_path, pkl_path,esm2_33_path,esm2_5120_path,ProtTrans_path,mas_path,residue=True,one_hot=True,esm2_33=True,esm_5120=True,prottrans=True,msa=True)
distance_matrixs=create_dis_matrix(dis_path,query_ids)

# X_train = X[:573]
X_test = X[573:]

# y_train = y[:573]
y_test = y[573:]


NUMBER_EPOCHS = 35

dr=0.3
lr=0.0001
nlayers=4
lamda=1.1
alpha=0.1
atten_time=8

# 构建五倍交叉验证适用的数据

IDs = query_ids[573:]

sequences = []
labels = []
with open(all_702_path,'r') as f:
    train_text = f.readlines()
    for i in range(0, len(train_text), 3):
        seq = train_text[i+1].strip()
        label = train_text[i+2].strip()
        sequences.append(seq)
        labels.append(label)

sequences = sequences[573:]


labels = y_test
features = X_test

coors = get_coor_test(dis_path, query_ids)
# use predicted structure to create new adj->graph->efeats
# adjs = get_adj_predicted(IDs)
adjs = get_adj_predicted(IDs)

graphs = []
for adj in adjs:
    edge_index, _ = dense_to_sparse(adj)
    G = dgl.graph((edge_index[0], edge_index[1])).to(device)
    graphs.append(G)

# # 先创建，后直接读取边特征
# print('start to prepare edge features')
# efeats = []
# for i in range(len(IDs)):
#     edge_feats = get_edge_attr_test(i,th=17,distance_matrixs=distance_matrixs)
#     print(edge_feats.shape)
#     efeats.append(edge_feats)
# save_edgefeats_path = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/predicted_structure/Train_Test129/edge_features'
# with open(save_edgefeats_path + '/EdgeFeats_prediceted_SC_17_129.pkl', 'wb') as f:
#     pickle.dump(efeats, f)

save_edgefeats_path = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/predicted_structure/Train_Test129/edge_features/EdgeFeats_prediceted_SC_17_129.pkl'
with open(save_edgefeats_path, 'rb') as f:
    efeats = pickle.load(f)

# print(len(efeats))

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
    embeddings = []

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
                efeat_batch = efeat_batch.to(device)
                adj_batch = Variable(adj_batch.cuda())
                coors_batch = Variable(coors_batch.cuda())
                y_true = label_batch
            else:
                node_features_batch = Variable(node_features_batch)
                graph_batch = graph_batch
                adj_batch = Variable(adj_batch)
                coors_batch = Variable(coors_batch)
                y_true = label_batch
                efeat_batch = efeat_batch

            y_pred = model( node_features_batch, adj_batch)
            # y_pred = model(graph_batch, node_features_batch,coors_batch,adj_batch,efeat_batch)  # ori
            # embeddings.append(embedding)

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
        best_aupr = 0
        best_threshold = 0

        for j in range(0, 100):

            threshold = j / 100000

            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            mcc = matthews_corrcoef(binary_true, binary_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold

        print(best_mcc)
        print(best_threshold)

    binary_pred = [1.0 if pred >= best_threshold else 0.0 for pred in y_pred]

    correct_samples = sum(a == b for a, b in zip(binary_pred, y_true))
    accuracy = correct_samples / len(y_true)

    pre = precision_score(y_true, binary_pred, zero_division=0)
    recall = recall_score(y_true, binary_pred, zero_division=0)
    f1 = f1_score(y_true, binary_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, binary_pred)

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
        'thred':best_threshold }

    return results


def test_129(Model_Path):

    # model = MainModel().to(device)

    # model = MainModel(dr,lr,nlayers,lamda,alpha,atten_time).to(device)  # ori
    model = MainModel(dr, lr, nlayers, lamda, alpha, atten_time, nfeats=71 + 20 + 256 + 1024 + 5120 + 33).to(device)

    model.load_state_dict(torch.load(Model_Path,map_location=device))

    test_dataSet = dataSet(dataframe=dataframe, adjs=adjs)
    print(test_dataSet.features.shape)
    test_loader = torch.utils.data.DataLoader(test_dataSet, batch_size=1, shuffle=False, collate_fn=graph_collate)

    _, test_true, test_pred = evaluate(model, test_loader)

    test_results = analysis(test_true, test_pred)

    return test_results


# trained_model save path
# Model_Path_1 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/AUPR_4_Predicted/Fold5predicted_edgeFeats_best_AUPR_model.pkl'  # ori

# Model_Path_1 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_EGNN/Fold1onlyEGNN_Predict_EdgeFeats_AUPR.pkl'  # for EGNN
# Model_Path_1 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_GCNII/Fold1_onlyGCNII4_Predict_EdgeFeats_AUPR.pkl'  # for EGNN
# Model_Path_2 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_GCNII/Fold2_onlyGCNII4_Predict_EdgeFeats_AUPR.pkl'  # for GCNII
# Model_Path_3 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_GCNII/Fold3_onlyGCNII4_Predict_EdgeFeats_AUPR.pkl'  # for GCNII
# Model_Path_4 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_GCNII/Fold4_onlyGCNII4_Predict_EdgeFeats_AUPR.pkl'  # for GCNII
# Model_Path_5 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ablation_study/model_ablation/only_GCNII/Fold5_onlyGCNII4_Predict_EdgeFeats_AUPR.pkl'  # for GCNII

# Model_Path_2 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/AUPR_4/Fold2_edgeFeats_best_AUPR_model.pkl'
# Model_Path_3 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/AUPR_4/Fold3_edgeFeats_best_AUPR_model.pkl'
# Model_Path_4 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/AUPR_4/Fold4_edgeFeats_best_AUPR_model.pkl'
# Model_Path_5 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/AUPR_4/Fold5_edgeFeats_best_AUPR_model.pkl'

Model_Path_1 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ComparedWith_GCNIIBiLSTM/Fold1_predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_2 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ComparedWith_GCNIIBiLSTM/Fold2_predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_3 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ComparedWith_GCNIIBiLSTM/Fold3_predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_4 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ComparedWith_GCNIIBiLSTM/Fold4_predicted_edgeFeats_best_AUPR_model.pkl'
Model_Path_5 = '/home/lichangyong/Documents/zmx/Graph_fusion/models/ComparedWith_GCNIIBiLSTM/Fold5_predicted_edgeFeats_best_AUPR_model.pkl'


# test_results_1,embeddings_1 = test_129(Model_Path_1)
# test_results_2,embeddings_2 = test_129(Model_Path_2)
# test_results_3,embeddings_3 = test_129(Model_Path_3)
# test_results_4,embeddings_4 = test_129(Model_Path_4)
# test_results_5,embeddings_5 = test_129(Model_Path_5)

test_results_1= test_129(Model_Path_1)
test_results_2= test_129(Model_Path_2)
test_results_3= test_129(Model_Path_3)
test_results_4= test_129(Model_Path_4)
test_results_5= test_129(Model_Path_5)

acc = (test_results_1['accuracy']+test_results_2['accuracy']+test_results_3['accuracy']+test_results_4['accuracy']+test_results_5['accuracy'])/5
spe = (test_results_1['spe']+test_results_2['spe']+test_results_3['spe']+test_results_4['spe']+test_results_5['spe'])/5
pre = (test_results_1['precision']+test_results_2['precision']+test_results_3['precision']+test_results_4['precision']+test_results_5['precision'])/5
recall = (test_results_1['recall']+test_results_2['recall']+test_results_3['recall']+test_results_4['recall']+test_results_5['recall'])/5
f1 = (test_results_1['f1']+test_results_2['f1']+test_results_3['f1']+test_results_4['f1']+test_results_5['f1'])/5
mcc = (test_results_1['mcc']+test_results_2['mcc']+test_results_3['mcc']+test_results_4['mcc']+test_results_5['mcc'])/5
auc = (test_results_1['auc']+test_results_2['auc']+test_results_3['auc']+test_results_4['auc']+test_results_5['auc'])/5
pr_auc = (test_results_1['pr_auc']+test_results_2['pr_auc']+test_results_3['pr_auc']+test_results_4['pr_auc']+test_results_5['pr_auc'])/5

# acc = (test_results_1['accuracy'])
# spe = (test_results_1['spe'])
# pre = (test_results_1['precision'])
# recall = (test_results_1['recall'])
# f1 = (test_results_1['f1'])
# mcc = (test_results_1['mcc'])
# auc = (test_results_1['auc'])
# pr_auc = (test_results_1['pr_auc'])


print("Test_129 performance on our method")
print("average accuracy: {} \n".format(acc))
print("average spe: {} \n".format(spe))
print("average pre: {} \n".format(pre))
print("average recall: {} \n".format(recall))
print("average f1: {} \n".format(f1))
print("average mcc: {} \n".format(mcc))
print("average auc: {} \n".format(auc))
print("average pr_auc: {} \n".format(pr_auc))

# file_for_draw = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/intermediate_files/Train_Test129/TSNE/'
# with open(file_for_draw + 'fold1_embedded.pkl', 'wb') as f:
#     pickle.dump(embeddings_1, f)
# with open(file_for_draw + 'fold2_embedded.pkl', 'wb') as f:
#     pickle.dump(embeddings_2, f)
# with open(file_for_draw + 'fold3_embedded.pkl', 'wb') as f:
#     pickle.dump(embeddings_3, f)
# with open(file_for_draw + 'fold4_embedded.pkl', 'wb') as f:
#     pickle.dump(embeddings_4, f)
# with open(file_for_draw + 'fold5_embedded.pkl', 'wb') as f:
#     pickle.dump(embeddings_5, f)