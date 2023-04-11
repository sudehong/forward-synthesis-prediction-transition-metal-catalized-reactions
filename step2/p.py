import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)

from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = True
from rdkit import Chem
import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)
import pandas as pd
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = True
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import  nn
import os
from tqdm import tqdm

data = pd.read_csv('NCI1_dataset_2.csv')
y1 = data.iloc[:, 1]

le = LabelEncoder()
label = le.fit_transform(y1)

smiles = data.iloc[:, 0]

s = smiles.tolist()


def f(xx):
    smi0 = xx
    smi1 = Chem.MolToSmiles(Chem.MolFromSmiles(smi0))  # normalize smiles
    m1 = Chem.MolFromSmiles(smi1)
    m1H = Chem.AddHs(m1)

    # 设置学习参数
    H_es, H_ep = 1, 1
    C_es, C_ep = 1, 1
    N_es, N_ep = 1, 1
    O_es, O_ep = 1, 1
    F_es, F_ep = 1, 1
    Cl_es, Cl_ep = 1, 1
    Br_es, Br_ep = 1, 1
    I_es, I_ep = 1, 1

    # matrix of sp,sp2,sp3
    def get_matrix(s, p, label):
        # SP
        matrix = torch.tensor([])
        if label == 'SP':
            es, ep = s, p
            matrix = torch.tensor([[1 / 2 * es + 1 / 2 * ep, 1 / 2 * es - 1 / 2 * ep, 0, 0],
                                   [1 / 2 * es - 1 / 2 * ep, 1 / 2 * es + 1 / 2 * ep, 0, 0],
                                   [0, 0, ep, 0],
                                   [0, 0, 0, ep]
                                   ])
        # SP2
        if label == 'SP2':
            es, ep = s, p
            matrix = torch.tensor([[1 / 3 * es + 2 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 0],
                                   [1 / 3 * es - 1 / 3 * ep, 1 / 3 * es + 2 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 0],
                                   [1 / 3 * es - 1 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 1 / 3 * es + 2 / 3 * ep, 0],
                                   [0, 0, 0, ep]
                                   ])
        # SP3
        if label == 'SP3':
            es, ep = s, p
            matrix = torch.tensor(
                [[1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep],
                 [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep],
                 [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep],
                 [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep]
                 ])
        if label == 'UNSPECIFIED' or label == 'S':
            es = s
            matrix = torch.tensor([es])

        return matrix

    diag = []
    for a1 in m1H.GetAtoms():
        # print(a1.GetSymbol(),a1.GetIdx(),a1.GetAtomicNum(),a1.GetHybridization())
        names = locals()
        es = names['%s_es' % str(a1.GetSymbol())]
        ep = names['%s_ep' % str(a1.GetSymbol())]
        diag.append(get_matrix(es, ep, str(a1.GetHybridization())))

    c = torch.tensor([])
    for i, v in enumerate(diag):
        if i == 0:
            c = torch.block_diag(v)
        else:
            c = torch.block_diag(c, v)

    # find number
    def FindAtomsNumberOfH(id):
        num = 0
        for a1 in m1H.GetAtoms():
            # print(a1.GetSymbol(),a1.GetIdx(),a1.GetAtomicNum(),a1.GetHybridization())
            if str(a1.GetHybridization()) == 'SP':
                num += 4
                if a1.GetIdx() == id:
                    break
            elif str(a1.GetHybridization()) == 'SP2':
                num += 4
                if a1.GetIdx() == id:
                    break
            elif str(a1.GetHybridization()) == 'SP3':
                num += 4
                if a1.GetIdx() == id:
                    break
            elif str(a1.GetHybridization()) == 'SP3D2':
                num += 6
                if a1.GetIdx() == id:
                    break
            elif str(a1.GetHybridization()) == 'UNSPECIFIED' or str(a1.GetHybridization()) == 'S':
                num += 1
                if a1.GetIdx() == id:
                    break
        return num

    ccsigma = 1
    chsigma = 1
    ccpai = 1
    onsigma = 1
    onpi = 1
    cnsigama = 1

    cclsigama = 1
    cfsigama = 1

    cosigama = 1
    cisigama = 1
    nhsigama = 1
    nnsigama = 1

    ccpai_15 = 1
    cnpi_15 = 1

    # 为每一个 c 的sigama设置一个随机取值的查找表
    s = []
    for a1 in m1H.GetAtoms():
        if str(a1.GetHybridization()) == 'SP':
            C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
            s.append([C_id, C_id + 1])
        elif str(a1.GetHybridization()) == 'SP2':
            C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
            s.append([C_id, C_id + 1, C_id + 2])
        elif str(a1.GetHybridization()) == 'SP3':
            C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
            s.append([C_id, C_id + 1, C_id + 2, C_id + 3])
        # elif str(a1.GetHybridization()) == 'SP3D2':
        #     C_id = FindAtomsNumberOfH(a1.GetIdx()) - 6
        #     s.append([C_id, C_id + 1, C_id + 2, C_id + 3, C_id + 4, C_id + 5])
        elif str(a1.GetHybridization()) == 'UNSPECIFIED' or str(a1.GetHybridization()) == 'S':
            C_id = FindAtomsNumberOfH(a1.GetIdx()) - 1
            s.append([C_id])
    # 为每一个 c 的pi设置一个随机取值的查找表
    p = []
    for a1 in m1H.GetAtoms():
        if str(a1.GetHybridization()) == 'SP':
            C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
            p.append([C_id + 2, C_id + 3, C_id + 3])
        elif str(a1.GetHybridization()) == 'SP2':
            C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
            # AROMATIC(2pi),C#C-C=C , N(=O)O-...(3pi)
            p.append([C_id + 3, C_id + 3, C_id + 3])
        elif str(a1.GetHybridization()) == 'SP3':
            C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
            p.append([])
        # elif str(a1.GetHybridization()) == 'SP3D2':
        #     C_id = FindAtomsNumberOfH(a1.GetIdx()) - 6
        #     p.append([])
        elif str(a1.GetHybridization()) == 'UNSPECIFIED' or str(a1.GetHybridization()) == 'S':
            p.append([])


    # 记录遍历的相邻的两个节点
    d = []
    for a1 in m1H.GetAtoms():
        for a2 in a1.GetNeighbors():
            id1 = a1.GetIdx()
            id2 = a2.GetIdx()
            # 处理重复遍历边问题
            if int(id1) > int(id2):
                id_of_edge = (id2, id1)
            else:
                id_of_edge = (id1, id2)

            # id_of_two_nodes = id1 + id2
            if id_of_edge in d:
                continue
            else:
                d.append(id_of_edge)

            # 判断键类型，并在矩阵中连接
            b12 = m1H.GetBondBetweenAtoms(a1.GetIdx(), a2.GetIdx())
            if b12.GetBondTypeAsDouble() == 1:
                # 如果是单链，直接sigama连接
                # 判断连接的两个元素 O-N
                if (str(a1.GetSymbol()) == 'O' and str(a2.GetSymbol()) == 'N') or (
                        str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'O'):
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = onsigma
                    # N(=O)O-,add pi
                    if str(a1.GetSymbol()) == 'N':
                        for a3 in a1.GetNeighbors():
                            #  find N=O
                            b13 = m1H.GetBondBetweenAtoms(a1.GetIdx(), a3.GetIdx())
                            if str(a3.GetSymbol()) == 'O' and str(b13.GetBondType()) == 'DOUBLE':
                                # add pi of (O-N)
                                C1 = p[a1.GetIdx()].pop()
                                C2 = p[a2.GetIdx()].pop()
                                c[C1, C2] = c[C2, C1] = onpi
                    elif str(a2.GetSymbol()) == 'N':
                        for a3 in a2.GetNeighbors():
                            #  find N=O
                            b23 = m1H.GetBondBetweenAtoms(a2.GetIdx(), a3.GetIdx())
                            if str(a3.GetSymbol()) == 'O' and str(b23.GetBondType()) == 'DOUBLE':
                                # add pi of (O-N)
                                C1 = p[a1.GetIdx()].pop()
                                C2 = p[a2.GetIdx()].pop()
                                c[C1, C2] = c[C2, C1] = onpi

                # C-H
                if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'H') or (
                        str(a1.GetSymbol()) == 'H' and str(a2.GetSymbol()) == 'C'):
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = chsigma

                # C-C
                if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'C':
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = ccsigma
                # C-N
                if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'N') or (
                        str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'C'):
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = cnsigama

                # c-cl
                if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'Cl') or (
                        str(a1.GetSymbol()) == 'Cl' and str(a2.GetSymbol()) == 'C'):
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = cclsigama

                # c-f
                if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'F') or (
                        str(a1.GetSymbol()) == 'F' and str(a2.GetSymbol()) == 'C'):
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = cfsigama

                # c-o
                if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'O') or (
                        str(a1.GetSymbol()) == 'O' and str(a2.GetSymbol()) == 'C'):
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = cosigama

                # c-i
                if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'I') or (
                        str(a1.GetSymbol()) == 'I' and str(a2.GetSymbol()) == 'C'):
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = cisigama

                # N-H
                if (str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'H') or (
                        str(a1.GetSymbol()) == 'H' and str(a2.GetSymbol()) == 'N'):
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = nhsigama

                # N-N
                if str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'N':
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = nnsigama

            # 如果是双链
            if b12.GetBondTypeAsDouble() == 1.5 or b12.GetBondTypeAsDouble() == 2:
                # O = N
                if (str(a1.GetSymbol()) == 'O' and str(a2.GetSymbol()) == 'N') or (
                        str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'O'):
                    # 一个sigama
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = onsigma

                    # 一个pi
                    C1 = p[a1.GetIdx()].pop()
                    C2 = p[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = onpi

                # C=C(double)
                # C=C
                if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'C':
                    # 一个sigama
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = ccsigma
                    ## 一个pi(, AROMATIC  or double)
                    if b12.GetBondTypeAsDouble() == 1.5:
                        # print(smi0,a1.GetIdx(),a2.GetIdx())
                        C1 = p[a1.GetIdx()].pop()
                        C2 = p[a2.GetIdx()].pop()
                        c[C1, C2] = c[C2, C1] = ccpai_15
                    else:
                        C1 = p[a1.GetIdx()].pop()
                        C2 = p[a2.GetIdx()].pop()
                        c[C1, C2] = c[C2, C1] = ccpai

                # C=N(AROMATIC,1.5)
                if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'N') or (
                        str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'C'):
                    # 一个sigama
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = cnsigama

                    # 一个pi
                    C1 = p[a1.GetIdx()].pop()
                    C2 = p[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = cnpi_15

            # 如果是三链
            if b12.GetBondTypeAsDouble() == 3:
                # C=C
                if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'C':
                    # C=C-C#C
                    # 一个sigama
                    C1 = s[a1.GetIdx()].pop()
                    C2 = s[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = ccsigma
                    # 第一个pi
                    C1 = p[a1.GetIdx()].pop()
                    C2 = p[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = ccpai
                    # 第二个pi
                    C1 = p[a1.GetIdx()].pop()
                    C2 = p[a2.GetIdx()].pop()
                    c[C1, C2] = c[C2, C1] = ccpai

                    # --C=C-C#C-C=C--
                    # a2位置
                    for a3 in a2.GetNeighbors():
                        a23 = m1H.GetBondBetweenAtoms(a2.GetIdx(), a3.GetIdx())
                        if str(a3.GetSymbol()) == 'C' and a23.GetBondTypeAsDouble() == 1:
                            for a4 in a3.GetNeighbors():
                                b34 = m1H.GetBondBetweenAtoms(a3.GetIdx(), a4.GetIdx())
                                if str(a4.GetSymbol()) == 'C' and b34.GetBondTypeAsDouble() == 2:
                                    # a3,a2额外一个pi连接
                                    C1 = p[a2.GetIdx()].pop()
                                    C2 = p[a3.GetIdx()].pop()
                                    c[C1, C2] = c[C2, C1] = ccpai

                    # a1位置
                    for a3 in a1.GetNeighbors():
                        a13 = m1H.GetBondBetweenAtoms(a1.GetIdx(), a3.GetIdx())
                        if str(a3.GetSymbol()) == 'C' and a13.GetBondTypeAsDouble() == 1:
                            for a4 in a3.GetNeighbors():
                                b34 = m1H.GetBondBetweenAtoms(a3.GetIdx(), a4.GetIdx())
                                if str(a4.GetSymbol()) == 'C' and b34.GetBondTypeAsDouble() == 2:
                                    # a3,a4额外一个pi连接
                                    C1 = p[a1.GetIdx()].pop()
                                    C2 = p[a3.GetIdx()].pop()
                                    c[C1, C2] = c[C2, C1] = ccpai

    emo, cmo = torch.linalg.eigh(c)
    # 求N
    N = 0
    for a1 in m1H.GetAtoms():
        if str(a1.GetSymbol()) == 'C' or str(a1.GetSymbol()) == 'N' or str(a1.GetSymbol()) == 'O' or str(
                a1.GetSymbol()) == 'F':
            N += int(a1.GetAtomicNum()) - 2
            # print(a1.GetSymbol(), a1.GetIdx(), a1.GetAtomicNum(), a1.GetHybridization())
        if str(a1.GetSymbol()) == 'Cl' or str(a1.GetSymbol()) == 'Br' or str(a1.GetSymbol()) == 'I':
            N += 7
        if str(a1.GetSymbol()) == 'H':
            N += 1
    n = N // 2
    x = emo[n - 2:n + 4]
    return x



for i in tqdm(s):
    x = f(i)
    if len(x) != 6 :
        print(i)
    # try:
    #     x  = f(i)
    # except IndexError:
    #     print(i)
    #     sum += 1
    #     continue
