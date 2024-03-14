# 2023.12.12
# create edges

import pickle
import numpy as np
import torch
import torch.nn as nn

def create_dis_matrix(dis_path,Query_ids):

    # calculate Euclidean distance then creating distance matrix

    dis_load=open(dis_path,'rb')
    dis_residue=pickle.load(dis_load)


    # distance_matrixs load all_702 proteins' distance matrix
    distance_matrixs=[]

    for i in Query_ids:
        residues=dis_residue[i]
        num_node = len(residues)
        residues_array = np.array(residues)

        distance_matrix = np.zeros((num_node, num_node))

        distances = np.linalg.norm(residues_array[:, np.newaxis, :] - residues_array[np.newaxis, :, :], axis=-1)
        distance_matrix[np.triu_indices(num_node, k=1)] = distances[np.triu_indices(num_node, k=1)]
        distance_matrix += distance_matrix.T        # undirected graphs, so the distance matrix is symmetrical

        distance_matrixs.append(distance_matrix)

    return distance_matrixs

# 获得一条蛋白质的边索引
def cal_edges(dis_matrix,protein_idx,th):
    # 创建一个布尔掩码，保留距离矩阵中值为1的边

    dis_matrix_copy = dis_matrix.copy()

    binary_matrix = (dis_matrix_copy[protein_idx] <= th).astype(int)
    symmetric_matrix = np.triu(binary_matrix) + np.triu(binary_matrix, 1).T
    dis_matrix_copy[protein_idx] = symmetric_matrix
    binary_matrix = torch.from_numpy(dis_matrix_copy[protein_idx])

    mask = (binary_matrix ==1)

    # 找到存在的边的索引
    radius_index_list = np.where(mask)

    # 转换为嵌套列表的形式
    radius_index_list = [list(nodes) for nodes in zip(radius_index_list[0], radius_index_list[1])]

    return radius_index_list


# 计算每条边上的两个属性，cosine of angle and 3D distance
def calculate_edge_attributes(edge_index_list, distance_matrixs, protein_idx):
    pdist = nn.PairwiseDistance(p=2, keepdim=True)
    cossim = nn.CosineSimilarity(dim=0)

    num_edges = len(edge_index_list)
    distance_attrs = torch.zeros(num_edges)
    cos_similarity_attrs = torch.zeros(num_edges)

    for i in range(num_edges):
        src_idx, dst_idx = edge_index_list[i]

        # 使用矩阵操作计算欧几里得距离
        distance_matrix_src = torch.tensor(distance_matrixs[protein_idx][src_idx][0])
        distance_matrix_dst = torch.tensor(distance_matrixs[protein_idx][dst_idx][0])
        distance = pdist(distance_matrix_src, distance_matrix_dst).item()
        distance_attrs[i] = distance / 17

        # 使用矩阵操作计算余弦相似度
        distance_matrix_src_array = torch.tensor(distance_matrixs[protein_idx][src_idx])
        distance_matrix_dst_array = torch.tensor(distance_matrixs[protein_idx][dst_idx])
        cos_similarity = cossim(distance_matrix_src_array, distance_matrix_dst_array).item()
        cos_similarity_attrs[i] = (cos_similarity + 1) / 2

    return distance_attrs, cos_similarity_attrs


def get_edge_attr_test_181(pro_id,th,distance_matrixs):

    edge_index_list = cal_edges(distance_matrixs, protein_idx=pro_id, th=th)
    distance_attrs, cos_similarity_attrs = calculate_edge_attributes(edge_index_list, distance_matrixs, protein_idx=pro_id)
    edge_attr_test = torch.stack((distance_attrs, cos_similarity_attrs), dim=1)

    return edge_attr_test