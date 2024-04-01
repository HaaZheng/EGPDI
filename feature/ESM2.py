import os, argparse
import sys
import numpy as np
from transformers import AutoTokenizer, EsmForMaskedLM
import torch

def ESM2(file):
    with open(file, 'r') as file:
        lines = file.readlines()
    
    model_file = "facebook/esm2_t36_3B_UR50D"      # create 33-dim embeddings
    # model_file = "facebook/esm2_t36_15B_UR50D"      # create 5120-dim embeddings

    tokenizer = AutoTokenizer.from_pretrained(model_file)
    model = EsmForMaskedLM.from_pretrained(model_file)

    for i in range(0, len(lines), 3):
        protein_id = lines[i].strip()[1:]
        sequence = lines[i + 1].strip()
        label = lines[i + 2].strip()
        lenn = len(label)
        
        seq = ""
        for i in range(lenn):  
            seq = seq + sequence[i] + " "

        inputs = tokenizer(seq, return_tensors="pt")
        with torch.no_grad():
            embeddings = model(**inputs).logits
        embeddings = embeddings[0][1:-1].detach().cpu().numpy()  
        print(np.array(embeddings).shape)

        np.save(protein_id + '.npy', embeddings)

path = 'Datasets/predicted_files/Train_Test129/DNA-573_Train.txt'
ESM2(path)
