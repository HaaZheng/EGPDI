
import os
import time
import pickle
from torch.utils.data import Dataset
from sklearn.metrics import matthews_corrcoef,  precision_score, recall_score, f1_score,roc_auc_score, average_precision_score,confusion_matrix
from torch_geometric.utils import dense_to_sparse
from sklearn.model_selection import KFold
import dgl
import torch
import pandas as pd
from torch.autograd import Variable
from torch.utils import data

from model_with_edge_features import MainModel
from feature.create_node_feature import create_dataset
from feature.create_graphs import get_coor_train,get_adj_predicted
from feature.create_edge import create_dis_matrix,get_edge_attr_train

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

import warnings
warnings.filterwarnings("ignore")
seed_value = 1995
th=17

# predicted structure
Model_Path = '/home/lichangyong/Documents/zmx/Graph_fusion/models/AUPR_4_Predicted/'

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

# predicted structure
root_dir = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/predicted_structure/Train_Test129/'
train_path= root_dir + 'DNA-573_Train.txt'
test_path= root_dir + 'DNA-129_Test.txt'
all_702_path = root_dir +  'DNA-702.txt'
pkl_path= root_dir + 'PDNA_residue_feas_PHSA.pkl'
esm2_5120_path= root_dir + 'ESM2-t48/'
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
print(len(query_ids))

X,y = create_dataset(query_ids,train_path, test_path,all_702_path, pkl_path,esm2_33_path,esm2_5120_path,ProtTrans_path,mas_path,residue=True,one_hot=True,esm2_33=True,esm_5120=True,prottrans=True,msa=True)

distance_matrixs=create_dis_matrix(dis_path,query_ids)
print(len(distance_matrixs))

X_train = X[:573]
X_test = X[573:]

y_train = y[:573]
y_test = y[573:]


NUMBER_EPOCHS = 30

# final model parameters
# dr=0.3,lr=0.0001,nlayers=4,lamda=1.1,alpha=0.1,atten_time=8, th trained on mcc and /100000 and saved bestAUPR model
dr=0.3
lr=0.0001
nlayers=4
lamda=1.1
alpha=0.1
atten_time=8

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

# use predicted structure to create new adj->graph->efeats
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
#     edge_feats = get_edge_attr_train(i,th=17,distance_matrixs=distance_matrixs)
#     print(edge_feats.shape)
#     efeats.append(edge_feats)
# save_edgefeats_path = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/predicted_structure/Train_Test129/edge_features'
# with open(save_edgefeats_path + '/EdgeFeats_predicted_SC_17.pkl', 'wb') as f:
#     pickle.dump(efeats, f)

# 573

save_edgefeats_path = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/predicted_structure/Train_Test129/edge_features/EdgeFeats_predicted_SC_17.pkl'
with open(save_edgefeats_path, 'rb') as f:
    efeats = pickle.load(f)

train_dic = {"ID": IDs, "sequence": sequences, "label": labels,'features':features,'coors':coors,'adj':adjs,'graph':graphs,'efeats':efeats}
dataframe = pd.DataFrame(train_dic)


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



def train_one_epoch(model,data_loader):

    epoch_loss_train = 0.0
    n = 0

    for label_batch, node_features_batch, graph_batch,efeat_batch,adj_batch,coors_batch in data_loader:

        model.optimizer.zero_grad()

        # sequence_name,sequence,label,node_features,graph,adj,coors
        node_features_batch = torch.tensor(node_features_batch)
        coors_batch = torch.tensor(coors_batch)
        adj_batch = adj_batch[0]
        label_batch = label_batch[0]
        efeat_batch = efeat_batch[0]

        # if torch.cuda.is_available():
        #     node_features_batch = Variable(node_features_batch.cuda())
        #     graph_batch = graph_batch.to(device)
        #     efeat_batch = efeat_batch.to(device)
        #     adj_batch = Variable(adj_batch.cuda())
        #     coors_batch = Variable(coors_batch.cuda())
        #     y_true = label_batch
        # else:
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

        y_pred = model(graph_batch, node_features_batch,coors_batch,adj_batch,efeat_batch)
        y_pred = torch.squeeze(y_pred)
        y_pred = torch.sigmoid(y_pred)

        # true labels
        y_true_int = [int(label) for label in y_true]
        labels = torch.tensor(y_true_int,dtype=torch.float32,device=device)

        loss = model.criterion(y_pred, labels)
        loss.backward()  # backward gradient

        model.optimizer.step()  # update all parameters

        epoch_loss_train += loss.item()
        n += 1

    print('training time',n)
    epoch_loss_train_avg = epoch_loss_train / n

    return epoch_loss_train_avg


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

            # if torch.cuda.is_available():
            #     node_features_batch = Variable(node_features_batch.cuda())
            #     graph_batch = graph_batch.to(device)
            #     efeat_batch = efeat_batch.to(device)
            #     adj_batch = Variable(adj_batch.cuda())
            #     coors_batch = Variable(coors_batch.cuda())
            #     y_true = label_batch
            # else:
            node_features_batch = Variable(node_features_batch)
            graph_batch = graph_batch
            adj_batch = Variable(adj_batch)
            coors_batch = Variable(coors_batch)
            y_true = label_batch
            efeat_batch = efeat_batch

            y_pred = model(graph_batch, node_features_batch, coors_batch, adj_batch, efeat_batch)

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
        best_threshold = 0

        for j in range(0, 100):

            threshold = j / 100000  # pls change this threshold according to your code

            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            mcc = matthews_corrcoef(binary_true, binary_pred)

            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold
    print('best_threshold',best_threshold)
    binary_pred = [1.0 if pred >= best_threshold else 0.0 for pred in y_pred]

    # correct_samples = (binary_pred == y_true).sum().item()
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
        'thred':best_threshold
    }

    return results


def train_1(model,train_dataframe,valid_dataframe,fold = 0):

    train_dataSet = dataSet(dataframe=train_dataframe, adjs=adjs)
    train_loader = torch.utils.data.DataLoader(train_dataSet,batch_size=1,shuffle=True,collate_fn=graph_collate)

    valid_dataSet = dataSet(dataframe=valid_dataframe, adjs=adjs)
    valid_loader = torch.utils.data.DataLoader(valid_dataSet, batch_size=1, shuffle=True, collate_fn=graph_collate)

    best_epoch = 0
    best_val_acc = 0
    best_val_spe = 0
    best_val_pre = 0
    best_val_recall = 0
    best_val_f1 = 0
    best_val_mcc = 0
    best_val_auc = 0
    best_val_prauc = 0

    for epoch in range(NUMBER_EPOCHS):

        print("\n========== Train epoch " + str(epoch + 1) + " ==========")
        model.train()

        begin_time = time.time()
        epoch_loss_train_avg = train_one_epoch(model, train_loader)
        end_time = time.time()
        run_time = end_time - begin_time

        print("========== Evaluate Valid set ==========")
        epoch_loss_valid_avg, valid_true, valid_pred = evaluate(model, valid_loader)
        valid_results = analysis(valid_true, valid_pred)
        print("Valid loss: ", epoch_loss_valid_avg)
        print("Valid accuracy: ", valid_results['accuracy'])
        print("Valid spe: ", valid_results['spe'])
        print("Valid precision: ", valid_results['precision'])
        print("Valid recall: ", valid_results['recall'])
        print("Valid f1: ", valid_results['f1'])
        print("Valid mcc: ", valid_results['mcc'])
        print("Valid auc: ", valid_results['auc'])
        print("Valid pr_auc: ", valid_results['pr_auc'])
        print("Running Time: ", run_time)

        # record_path = '/home/lichangyong/Documents/zmx/Graph_fusion/record/' + 'fold' + str(fold) + '.txt'
        # with open(record_path, 'a') as f:
        #     f.write('epoch' + str(epoch + 1) + '\n')
        #     f.write("========== Evaluate Valid set ========== \n")
        #     f.write("Valid loss: {} \n".format(epoch_loss_valid_avg))
        #     f.write("Valid accuracy: {} \n".format(valid_results['accuracy']))
        #     f.write("Valid spe: {} \n".format(valid_results['spe']))
        #     f.write("Valid precision: {} \n".format(valid_results['precision']))
        #     f.write("Valid recall: {} \n".format(valid_results['recall']))
        #     f.write("Valid f1: {} \n".format(valid_results['f1']))
        #     f.write("Valid mcc: {} \n".format(valid_results['mcc']))
        #     f.write("Valid auc: {} \n".format(valid_results['auc']))
        #     f.write("Valid pr_auc: {} \n".format(valid_results['pr_auc']))
        #     f.write("Run time: {} \n".format(run_time))
        #     f.write('\n')

        if best_val_prauc < valid_results['pr_auc']:
            best_epoch = epoch + 1
            best_val_mcc = valid_results['mcc']
            best_val_acc = valid_results['accuracy']
            best_val_spe = valid_results['spe']
            best_val_pre = valid_results['precision']
            best_val_recall = valid_results['recall']
            best_val_f1 = valid_results['f1']
            best_val_auc = valid_results['auc']
            best_val_prauc = valid_results['pr_auc']

            print('-' * 20, "new best pr_auc:{0}".format(best_val_prauc), '-' * 20)
            # torch.save(model.state_dict(), os.path.join(Model_Path, 'Fold' + str(fold) + 'predicted_edgeFeats_best_AUPR_model.pkl'))

    return best_epoch,best_val_mcc,best_val_acc,best_val_spe,best_val_pre,best_val_recall,best_val_f1,best_val_auc,best_val_prauc


def cross_validation(all_dataframe,fold_number = 5):

    sequence_names = all_dataframe['ID'].values
    sequence_labels = all_dataframe['label'].values
    kfold = KFold(n_splits=fold_number, shuffle=True)
    fold = 0

    best_epochs = []
    valid_accs = []
    valid_spes = []
    valid_recalls = []
    valid_mccs = []
    valid_f1s = []
    valid_pres = []
    valid_aucs = []
    valid_pr_aucs = []

    for train_index,valid_index in kfold.split(sequence_names,sequence_labels):

        print("\n\n========== Fold " + str(fold + 1) + " ==========")
        train_dataframe = all_dataframe.iloc[train_index, :]
        valid_dataframe = all_dataframe.iloc[valid_index, :]
        print("Train on", str(train_dataframe.shape[0]),"samples, validate on",str(valid_dataframe.shape[0]),"samples")

        model = MainModel(dr,lr,nlayers,lamda,alpha,atten_time,nfeats=71+20+33+256+1024+5120)

        # if torch.cuda.is_available():
        #     model.cuda()

        best_epoch,valid_mcc,val_acc,val_spe,val_pre,val_recall,val_f1,val_auc,val_pr_auc = train_1(model,train_dataframe,valid_dataframe,fold+1)
        best_epochs.append(str(best_epoch))
        valid_mccs.append(valid_mcc)
        valid_accs.append(val_acc)
        valid_spes.append(val_spe)
        valid_pres.append(val_pre)
        valid_recalls.append(val_recall)
        valid_f1s.append(val_f1)
        valid_aucs.append(val_auc)
        valid_pr_aucs.append(val_pr_auc)

        fold += 1

    print("\n\nBest epoch: " + " ".join(best_epochs))
    print("Average MCC of {} fold：{:.4f}".format(fold_number, sum(valid_mccs) / fold_number))
    print("Average acc of {} fold：{:.4f}".format(fold_number, sum(valid_accs) / fold_number))
    print("Average spe of {} fold：{:.4f}".format(fold_number, sum(valid_spes) / fold_number))
    print("Average pre of {} fold：{:.4f}".format(fold_number, sum(valid_pres) / fold_number))
    print("Average recall of {} fold：{:.4f}".format(fold_number, sum(valid_recalls) / fold_number))
    print("Average f1 of {} fold：{:.4f}".format(fold_number, sum(valid_f1s) / fold_number))
    print("Average auc of {} fold：{:.4f}".format(fold_number, sum(valid_aucs) / fold_number))
    print("Average pr_auc of {} fold：{:.4f}".format(fold_number, sum(valid_pr_aucs) / fold_number))

    return round(sum([int(epoch) for epoch in best_epochs]) / fold_number)


aver_epoch = cross_validation(dataframe, fold_number = 5)
