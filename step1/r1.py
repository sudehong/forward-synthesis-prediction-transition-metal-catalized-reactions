import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np

f =  open("data.txt",'r')
lines = f.readlines()
x = []
Y = []

for line in lines:
    l = line.split(  )
    x.append(l[11])
    t = []
    for i in l[1:11]:
        t.append(float(i))
    Y.append(t)

print(x)
print(Y)

from rdkit import Chem

# from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = True
import numpy as np
import pandas as pd
import scipy.linalg
import random

class MyModel(torch.nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()
    self.w = torch.nn.Parameter(torch.rand(27,1,requires_grad =True, dtype=torch.float32))

  def forward(self,x):

      smi0 = x
      smi1 = Chem.MolToSmiles(Chem.MolFromSmiles(smi0))  # normalize smiles
      m1 = Chem.MolFromSmiles(smi1)
      m1H = Chem.AddHs(m1)

      # 设置学习参数
      H_es = self.w[0]
      C_es, C_ep = self.w[1],self.w[2]
      N_es, N_ep = self.w[3],self.w[4]
      O_es, O_ep = self.w[5],self.w[6]
      F_es, F_ep = self.w[7],self.w[8]
      Cl_es, Cl_ep = self.w[9],self.w[10]
      Br_es, Br_ep = self.w[11],self.w[12]
      I_es, I_ep = self.w[13],self.w[14]

      diag = []
      for a1 in m1H.GetAtoms():
          # print(a1.GetSymbol(),a1.GetIdx(),a1.GetAtomicNum(),a1.GetHybridization())
          if str(a1.GetHybridization()) == 'SP':
              if str(a1.GetSymbol()) == 'C':
                  es, ep = C_es, C_ep
                  SP = torch.tensor([[1 / 2 * es + 1 / 2 * ep, 1 / 2 * es - 1 / 2 * ep, 0, 0],
                                     [1 / 2 * es - 1 / 2 * ep, 1 / 2 * es + 1 / 2 * ep, 0, 0],
                                     [0, 0, ep, 0],
                                     [0, 0, 0, ep]
                                     ])
                  diag.append(SP)
              if str(a1.GetSymbol()) == 'N':
                  es, ep = N_es, N_ep
                  SP = torch.tensor([[1 / 2 * es + 1 / 2 * ep, 1 / 2 * es - 1 / 2 * ep, 0, 0],
                                     [1 / 2 * es - 1 / 2 * ep, 1 / 2 * es + 1 / 2 * ep, 0, 0],
                                     [0, 0, ep, 0],
                                     [0, 0, 0, ep]
                                     ])
                  diag.append(SP)
              # diag.append(SP)
          elif str(a1.GetHybridization()) == 'SP2':
              if str(a1.GetSymbol()) == 'C':
                  es, ep = C_es, C_ep
                  SP2 = torch.tensor([[1 / 3 * es + 2 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 0],
                                      [1 / 3 * es - 1 / 3 * ep, 1 / 3 * es + 2 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 0],
                                      [1 / 3 * es - 1 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 1 / 3 * es + 2 / 3 * ep, 0],
                                      [0, 0, 0, ep]
                                      ])
                  diag.append(SP2)
              if str(a1.GetSymbol()) == 'O':
                  es, ep = O_es, O_ep
                  SP2 = torch.tensor([[1 / 3 * es + 2 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 0],
                                      [1 / 3 * es - 1 / 3 * ep, 1 / 3 * es + 2 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 0],
                                      [1 / 3 * es - 1 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 1 / 3 * es + 2 / 3 * ep, 0],
                                      [0, 0, 0, ep]
                                      ])
                  diag.append(SP2)
              if str(a1.GetSymbol()) == 'N':
                  es, ep = N_es, N_ep
                  SP2 = torch.tensor([[1 / 3 * es + 2 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 0],
                                      [1 / 3 * es - 1 / 3 * ep, 1 / 3 * es + 2 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 0],
                                      [1 / 3 * es - 1 / 3 * ep, 1 / 3 * es - 1 / 3 * ep, 1 / 3 * es + 2 / 3 * ep, 0],
                                      [0, 0, 0, ep]
                                      ])
                  diag.append(SP2)

          elif str(a1.GetHybridization()) == 'SP3':
              if str(a1.GetSymbol()) == 'O':
                  es, ep = O_es, O_ep
                  SP3 = torch.tensor([[1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es + 3 / 4 * ep]
                                      ])
                  diag.append(SP3)
              if str(a1.GetSymbol()) == 'C':
                  es, ep = C_es, C_ep
                  SP3 = torch.tensor([[1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es + 3 / 4 * ep]
                                      ])
                  diag.append(SP3)
              if str(a1.GetSymbol()) == 'F':
                  es, ep = F_es, F_ep
                  SP3 = torch.tensor([[1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es + 3 / 4 * ep]
                                      ])
                  diag.append(SP3)
              if str(a1.GetSymbol()) == 'Cl':
                  es, ep = Cl_es, Cl_ep
                  SP3 = torch.tensor([[1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es + 3 / 4 * ep]
                                      ])
                  diag.append(SP3)
              if str(a1.GetSymbol()) == 'Br':
                  es, ep = Br_es, Br_ep
                  SP3 = torch.tensor([[1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es + 3 / 4 * ep]
                                      ])
                  diag.append(SP3)
              if str(a1.GetSymbol()) == 'I':
                  es, ep = I_es, I_ep
                  SP3 = torch.tensor([[1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es + 3 / 4 * ep]
                                      ])
                  diag.append(SP3)
              if str(a1.GetSymbol()) == 'N':
                  es, ep = N_es, N_ep
                  SP3 = torch.tensor([[1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep,
                                       1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep,
                                       1 / 4 * es + 3 / 4 * ep]
                                      ])
                  diag.append(SP3)

          elif str(a1.GetHybridization()) == 'UNSPECIFIED':
              if str(a1.GetSymbol()) == 'H':
                  es = H_es
                  S = torch.tensor([es])
                  diag.append(S)
      # print(len(diag))
      c = torch.tensor([])
      for i, v in enumerate(diag):
          if i == 0:
              c = torch.block_diag(v)
          else:
              c = torch.block_diag(c, v)
      # print(c.shape)
      # print(smi0)

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
              elif str(a1.GetHybridization()) == 'UNSPECIFIED':
                  num += 1
                  if a1.GetIdx() == id:
                      break
          return num

      ccsigma = self.w[15]
      chsigma = self.w[16]
      ccpai = self.w[17]
      onsigma = self.w[18]
      onpi = self.w[19]
      cnsigama = self.w[20]

      cclsigama = self.w[21]
      cfsigama = self.w[22]

      cosigama = self.w[23]
      cisigama = self.w[24]
      nhsigama = self.w[25]
      nnsigama = self.w[26]

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
          elif str(a1.GetHybridization()) == 'SP3D2':
              C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
              s.append([C_id, C_id + 1, C_id + 2, C_id + 3, C_id + 4, C_id + 5])
          elif str(a1.GetHybridization()) == 'UNSPECIFIED':
              C_id = FindAtomsNumberOfH(a1.GetIdx()) - 1
              s.append([C_id])
      # 为每一个 c 的pi设置一个随机取值的查找表
      p = []
      for a1 in m1H.GetAtoms():
          if str(a1.GetHybridization()) == 'SP':
              C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
              p.append([C_id + 2, C_id + 3])
          elif str(a1.GetHybridization()) == 'SP2':
              C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
              p.append([C_id + 3])
          elif str(a1.GetHybridization()) == 'SP3':
              C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
              p.append([])
          elif str(a1.GetHybridization()) == 'SP3D2':
              C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
              p.append([])
          elif str(a1.GetHybridization()) == 'UNSPECIFIED':
              p.append([])

      # 记录遍历的相邻的两个节点
      d = []
      for a1 in m1H.GetAtoms():
          for a2 in a1.GetNeighbors():
              aa1 = a1.GetSymbol()
              aa2 = a2.GetSymbol()

              id1 = a1.GetIdx()
              id2 = a2.GetIdx()
              # 记录已经处理的相同元素
              id_of_two_nodes = id1 + id2
              if id_of_two_nodes in d:
                  continue
              else:
                  d.append(id_of_two_nodes)
              # 判断键类型，并在矩阵中连接
              b12 = m1H.GetBondBetweenAtoms(a1.GetIdx(), a2.GetIdx())
              if b12.GetBondTypeAsDouble() == 1:
                  # 如果是单链，直接sigama连接
                  # 判断连接的两个元素 O-N
                  if str(a1.GetSymbol()) == 'O' and str(a2.GetSymbol()) == 'N':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = onsigma
                  if str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'O':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = onsigma
                  # C-H
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'H':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = chsigma
                  if str(a1.GetSymbol()) == 'H' and str(a2.GetSymbol()) == 'C':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = chsigma
                  # C-C
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'C':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = ccsigma
                  # C-N
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'N':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cnsigama
                  if str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'C':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cnsigama
                  # c-cl
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'Cl':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cclsigama
                  if str(a1.GetSymbol()) == 'Cl' and str(a2.GetSymbol()) == 'C':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cclsigama
                  # c-f
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'F':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cfsigama
                  if str(a1.GetSymbol()) == 'F' and str(a2.GetSymbol()) == 'C':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cfsigama
                  # c-o
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'O':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cosigama
                  if str(a1.GetSymbol()) == 'O' and str(a2.GetSymbol()) == 'C':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cosigama
                  # c-i
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'I':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cisigama
                  if str(a1.GetSymbol()) == 'I' and str(a2.GetSymbol()) == 'C':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cisigama
                  #N-H
                  if str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'H':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = nhsigama
                  if str(a1.GetSymbol()) == 'H' and str(a2.GetSymbol()) == 'N':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = nhsigama

                  #N-N
                  if str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'N':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = nnsigama

              # 如果是双链 O=N
              if b12.GetBondTypeAsDouble() == 1.5 or b12.GetBondTypeAsDouble() == 2:
                  # O = N
                  # 一个sigama
                  if str(a1.GetSymbol()) == 'O' and str(a2.GetSymbol()) == 'N':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = onsigma
                  if str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'O':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = onsigma
                  # 一个pi
                  if str(a1.GetSymbol()) == 'O' and str(a2.GetSymbol()) == 'N':
                      C1 = p[a1.GetIdx()].pop()
                      C2 = p[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = onpi
                  if str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'O':
                      C1 = p[a1.GetIdx()].pop()
                      C2 = p[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = onpi

                  # C=C
                  # 一个sigama
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'C':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = ccsigma

                  # 一个pi
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'C':
                      emptylist = []
                      if p[a2.GetIdx()] != emptylist:
                          pass
                      else:
                          C1 = p[a1.GetIdx()].pop()
                          C2 = p[a2.GetIdx()].pop()
                          c[C1, C2] = c[C2, C1] = ccpai

              # 如果是三链
              if b12.GetBondTypeAsDouble() == 3:
                  # C=C
                  # 一个sigama
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'C':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = ccsigma

                  # 第一个pi
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'C':
                      C1 = p[a1.GetIdx()].pop()
                      C2 = p[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = ccpai

                  # 第二个pi
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'C':
                      C1 = p[a1.GetIdx()].pop()
                      C2 = p[a2.GetIdx()].pop()
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
      return emo[n-4:n+6]

#
num = 100
lr = 0.01
criterion = torch.nn.MSELoss()
model = MyModel()
# optimizer = torch.optim.SGD(model.parameters(),lr)
optimizer = torch.optim.Adam(model.parameters(),lr, weight_decay=0)

y = torch.tensor(Y,dtype=torch.float32)
y_pred = []

xx = []
yy = []

for j  in range(num):
  optimizer.zero_grad()
  for i in x:
      y_pred.append(model(i))
  y_p = torch.stack(y_pred)

  loss = 0
  for y1,y_p1 in zip(y,y_p):
      loss += criterion(y1,y_p1)

  loss.requires_grad_(True)
  loss.backward()
  print(f'iter{j},loss {loss}')
  optimizer.step()
  y_pred.clear()
  xx.append(j)
  yy.append(loss.detach().numpy())
print(f'final parameter: {model.w}')


import matplotlib.pyplot as plt
plt.plot(xx,yy)
plt.show()