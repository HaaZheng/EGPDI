from Bio import AlignIO

# 读取.a3m文件
alignment = AlignIO.read("/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/predicted_structure/Test_181/test_615ad_0.a3m",'fasta')

# 提取MSA特征
for record in alignment:
    # 提取序列名称和序列
    sequence_name = record.id
    sequence = str(record.seq)

    # 处理序列，提取你感兴趣的特征（例如，氨基酸组成）
    amino_acids = [aa for aa in sequence]

    # 打印序列名称和特征
    print("Sequence Name:", sequence_name)
    print("Amino Acids:", amino_acids)