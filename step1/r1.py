import numpy as np
import torch
np.random.seed(0)
torch.manual_seed(0)
import pandas as pd

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
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.drawOptions.addAtomIndices = True
from torch.optim import lr_scheduler
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions



class MyModel(torch.nn.Module):
  def __init__(self):
    super(MyModel, self).__init__()
    self.w = torch.nn.Parameter(torch.randn(28,1,requires_grad =True, dtype=torch.float32))

  def forward(self,x):

      smi0 = x
      smi1 = Chem.MolToSmiles(Chem.MolFromSmiles(smi0))  # normalize smiles
      m1 = Chem.MolFromSmiles(smi1)
      m1H = Chem.AddHs(m1)

      # 设置学习参数
      H_es,H_ep = self.w[0],torch.tensor(0,dtype=torch.float32)
      C_es, C_ep = self.w[1],self.w[2]
      N_es, N_ep = self.w[3],self.w[4]
      O_es, O_ep = self.w[5],self.w[6]
      F_es, F_ep = self.w[7],self.w[8]
      Cl_es, Cl_ep = self.w[9],self.w[10]
      Br_es, Br_ep = self.w[11],self.w[12]
      I_es, I_ep = self.w[13],self.w[14]

      #matrix of sp,sp2,sp3
      def get_matrix(s,p,label):
          #SP
          if label == 'SP':
              es, ep = s, p
              matrix = torch.tensor([[1 / 2 * es + 1 / 2 * ep, 1 / 2 * es - 1 / 2 * ep, 0, 0],
                                     [1 / 2 * es - 1 / 2 * ep, 1 / 2 * es + 1 / 2 * ep, 0, 0],
                                     [0, 0, ep, 0],
                                     [0, 0, 0, ep]
                                                 ])
          #SP2
          if label == 'SP2':
              es, ep = s, p
              matrix = torch.tensor([ [1 / 3 * es + 2 / 3 * ep,  1 / 3 * es - 1 / 3 * ep,  1 / 3 * es - 1 / 3 * ep, 0],
                                      [1 / 3 * es - 1 / 3 * ep,  1 / 3 * es + 2 / 3 * ep,  1 / 3 * es - 1 / 3 * ep, 0],
                                      [1 / 3 * es - 1 / 3 * ep,  1 / 3 * es - 1 / 3 * ep,  1 / 3 * es + 2 / 3 * ep, 0],
                                      [0, 0, 0, ep]
                                      ])
          #SP3
          if label == 'SP3':
              es, ep = s, p
              matrix = torch.tensor([ [1 / 4 * es + 3 / 4 * ep,  1 / 4 * es - 1 / 4 * ep,  1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep,  1 / 4 * es + 3 / 4 * ep,  1 / 4 * es - 1 / 4 * ep, 1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep,  1 / 4 * es - 1 / 4 * ep,  1 / 4 * es + 3 / 4 * ep, 1 / 4 * es - 1 / 4 * ep],
                                      [1 / 4 * es - 1 / 4 * ep,  1 / 4 * es - 1 / 4 * ep,  1 / 4 * es - 1 / 4 * ep, 1 / 4 * es + 3 / 4 * ep]
                                      ])
          if label == 'UNSPECIFIED':
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

      ccpai_15 = self.w[27]


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
              p.append([C_id + 2, C_id + 3,C_id + 3])
          elif str(a1.GetHybridization()) == 'SP2':
              C_id = FindAtomsNumberOfH(a1.GetIdx()) - 4
              #AROMATIC(2pi),C#C-C=C , N(=O)O-...(3pi)
              p.append([C_id + 3,C_id + 3,C_id + 3])
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
              id1 = a1.GetIdx()
              id2 = a2.GetIdx()
              #处理重复遍历边问题
              if int(id1) > int(id2):
                  id_of_edge = (id2,id1)
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
                  if (str(a1.GetSymbol()) == 'O' and str(a2.GetSymbol()) == 'N') or (str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'O'):
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = onsigma
                      #N(=O)O-,add pi
                      if str(a1.GetSymbol()) == 'N':
                          for  a3 in a1.GetNeighbors():
                              #  find N=O
                              b13 = m1H.GetBondBetweenAtoms(a1.GetIdx(), a3.GetIdx())
                              if str(a3.GetSymbol()) == 'O' and str(b13.GetBondType()) == 'DOUBLE':
                                  # add pi of (O-N)
                                  C1 = p[a1.GetIdx()].pop()
                                  C2 = p[a2.GetIdx()].pop()
                                  c[C1, C2] = c[C2, C1] = onpi
                      elif str(a2.GetSymbol()) == 'N':
                          for  a3 in a2.GetNeighbors():
                              #  find N=O
                              b23 = m1H.GetBondBetweenAtoms(a2.GetIdx(), a3.GetIdx())
                              if str(a3.GetSymbol()) == 'O' and str(b23.GetBondType()) == 'DOUBLE':
                                  # add pi of (O-N)
                                  C1 = p[a1.GetIdx()].pop()
                                  C2 = p[a2.GetIdx()].pop()
                                  c[C1, C2] = c[C2, C1] = onpi

                  # C-H
                  if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'H') or (str(a1.GetSymbol()) == 'H' and str(a2.GetSymbol()) == 'C'):
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = chsigma

                  # C-C
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'C':
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = ccsigma
                  # C-N
                  if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'N') or (str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'C'):
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cnsigama

                  # c-cl
                  if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'Cl') or (str(a1.GetSymbol()) == 'Cl' and str(a2.GetSymbol()) == 'C'):
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cclsigama

                  # c-f
                  if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'F') or (str(a1.GetSymbol()) == 'F' and str(a2.GetSymbol()) == 'C'):
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cfsigama

                  # c-o
                  if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'O') or (str(a1.GetSymbol()) == 'O' and str(a2.GetSymbol()) == 'C'):
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cosigama

                  # c-i
                  if (str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'I') or (str(a1.GetSymbol()) == 'I' and str(a2.GetSymbol()) == 'C'):
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = cisigama

                  #N-H
                  if (str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'H') or (str(a1.GetSymbol()) == 'H' and str(a2.GetSymbol()) == 'N'):
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

                  if (str(a1.GetSymbol()) == 'O' and str(a2.GetSymbol()) == 'N') or (str(a1.GetSymbol()) == 'N' and str(a2.GetSymbol()) == 'O'):
                      # 一个sigama
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = onsigma

                      # 一个pi
                      C1 = p[a1.GetIdx()].pop()
                      C2 = p[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = onpi

                  # C=C(double)
                  #C=C
                  if str(a1.GetSymbol()) == 'C' and str(a2.GetSymbol()) == 'C':
                      # 一个sigama
                      C1 = s[a1.GetIdx()].pop()
                      C2 = s[a2.GetIdx()].pop()
                      c[C1, C2] = c[C2, C1] = ccsigma
                      ## 一个pi(, AROMATIC  or double)
                      if b12.GetBondTypeAsDouble() == 1.5:
                          C1 = p[a1.GetIdx()].pop()
                          C2 = p[a2.GetIdx()].pop()
                          c[C1, C2] = c[C2, C1] = ccpai_15
                      else:
                          C1 = p[a1.GetIdx()].pop()
                          C2 = p[a2.GetIdx()].pop()
                          c[C1, C2] = c[C2, C1] = ccpai

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

                      #--C=C-C#C-C=C--
                      #a2位置
                      for a3 in a2.GetNeighbors():
                          if str(a3.GetSymbol()) == 'C':
                              for a4 in a2.GetNeighbors():
                                  b34 = m1H.GetBondBetweenAtoms(a3.GetIdx(), a4.GetIdx())
                                  if str(a4.GetSymbol()) == 'C' and b34.GetBondTypeAsDouble() == 2:
                                      # a3,a4额外一个pi连接
                                      C1 = p[a2.GetIdx()].pop()
                                      C2 = p[a3.GetIdx()].pop()
                                      c[C1, C2] = c[C2, C1] = ccpai
                      # a1位置
                      for a3 in a1.GetNeighbors():
                          if str(a3.GetSymbol()) == 'C':
                              for a4 in a2.GetNeighbors():
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
          if str(a1.GetSymbol()) == 'C' or str(a1.GetSymbol()) == 'N' or str(a1.GetSymbol()) == 'O' or str(a1.GetSymbol()) == 'F':
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
# scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
# scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

device=torch.device("cuda"if torch.cuda.is_available() else "cpu")

y = torch.tensor(Y,dtype=torch.float32)
y_pred = []

xx = []
yy = []

loss_sort = []

for j  in range(num):
  optimizer.zero_grad()
  for i in x:
      y_pred.append(model(i))
  y_p = torch.stack(y_pred)

  loss = 0
  for y1,y_p1,xx12 in zip(y,y_p,x):
      loss += criterion(y1,y_p1)
      if j == (num-1):loss_sort.append({'loss_of_each_mol':criterion(y1,y_p1).item(),'smi':xx12})

  loss.requires_grad_(True)
  loss.backward()
  print(f'iter{j},loss {loss}')
  optimizer.step()

  y_pred.clear()
  xx.append(j)
  yy.append(loss.detach().numpy())
# print(f'final parameter: {model.w}')

w = {'H_es':model.w[0]
     ,'C_es':model.w[1]
     ,'C_ep':model.w[2]
     ,'N_es':model.w[3],'N_ep':model.w[4]
     ,'O_es':model.w[5],'O_ep':model.w[6]
     ,'F_es':model.w[7],'F_ep':model.w[8]
     ,'Cl_es':model.w[9],'Cl_ep':model.w[10]
    ,'Br_es':model.w[11],'Br_ep':model.w[12]
    ,'I_es':model.w[13],'I_ep':model.w[14]
     ,'ccsigma' : model.w[15]
    ,'chsigma' : model.w[16]
    ,'ccpai' : model.w[17]
    ,'onsigma' : model.w[18]
    ,'onpi' : model.w[19]
    ,'cnsigama' : model.w[20]
    ,'cclsigama' : model.w[21]
    ,'cfsigama' : model.w[22]
    ,'cosigama' : model.w[23]
    ,'cisigama' : model.w[24]
    ,'nhsigama' : model.w[25]
    ,'nnsigama' : model.w[26] }

# w1 = sorted(w.items(), key=lambda x: x[1])
for keys,values in w.items():
    print(keys,values)

df = pd.DataFrame(loss_sort)
# df.reset_index(drop=True)

df.sort_values(by="loss_of_each_mol",inplace=True,ascending=False)
print(df)
print(df.head(5))
print(df.tail(5))

smis_head5 = df.head(5)['smi'].values.tolist()
mols = []
for smi in smis_head5:
    mol = Chem.MolFromSmiles(smi)
    mols.append(mol)

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=4,
    subImgSize=(700,700),
    legends=['' for x in mols]
    ,returnPNG=False
)
img.save('head5.jpg')

smis_tail5 = df.tail(5)['smi'].values.tolist()
mols = []
for smi in smis_tail5:
    mol = Chem.MolFromSmiles(smi)
    mols.append(mol)

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=4,
    subImgSize=(700,700),
    legends=['' for x in mols]
    ,returnPNG=False
)
img.save('tail5.jpg')

import matplotlib.pyplot as plt
plt.plot(xx,yy)
plt.xlabel('epcho')
plt.ylabel('loss')
plt.title('randn')
plt.show()