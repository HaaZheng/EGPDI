
import os, argparse
import sys
import numpy as np
from transformers import AutoTokenizer, EsmForMaskedLM
import torch


def fa2bert(file):
    with open(file, 'r') as file:
        lines = file.readlines()

    model_file = "facebook/esm2_t6_8M_UR50D"
    model_file = "facebook/esm2_t12_35M_UR50D"
    model_file = "facebook/esm2_t30_150M_UR50D"
    model_file = "facebook/esm2_t36_3B_UR50D"
    # model_file="facebook/esm2_t48_15B_UR50D"        #      太大，服务器内存带不起来

    tokenizer = AutoTokenizer.from_pretrained(model_file)
    model = EsmForMaskedLM.from_pretrained(model_file)

    for i in range(0, len(lines), 3):
        protein_id = lines[i].strip()[1:]
        sequence = lines[i + 1].strip()
        label = lines[i + 2].strip()

        lenn = len(label)
        seq = ""
        for i in range(lenn):  # 把每一个序列，整合为一个数组。
            seq = seq + sequence[i] + " "

        inputs = tokenizer(seq, return_tensors="pt")

        with torch.no_grad():
            embeddings = model(**inputs).logits

        embeddings = embeddings[0][1:-1].detach().cpu().numpy()  # 去除两边的两个令牌吧

        print(np.array(embeddings).shape)

        np.save("/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/raw_files/Test_181/ESM2-t36/" + protein_id + '.npy', embeddings)


def main():
    parser = argparse.ArgumentParser(description="deep learning 6mA analysis in rice genome")
    parser.add_argument("--path1", type=str, help="Train_335.fa", required=True)
    args = parser.parse_args()

    path1 = os.path.abspath(args.path1)
    # path1 = '/home/lichangyong/Documents/zmx/Graph_fusion/Datasets/raw_files/Test_181/DNA_Test_181.txt'

    if not os.path.exists(path1):
        print("The csv benchmark_data not exist! Error\n")
        sys.exit()
    fa2bert(path1)
    print("you did it")


if __name__ == "__main__":
    main()