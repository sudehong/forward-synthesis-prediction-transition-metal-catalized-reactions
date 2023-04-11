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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model1 import MyModel
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix,r2_score
import torch.nn.functional as F

from collections import Counter

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



class Mydateset(Dataset):
    def __init__(self,csv_path,transform=None):
        data = pd.read_csv(csv_path)
        y1 = data['activity']

        le = LabelEncoder()
        label = le.fit_transform(y1)

        smiles = data['smile']
        # smi_list = list(smiles)

        self.data = smiles.tolist()
        self.labels = label.tolist()

    def __getitem__(self, index):
        """サンプルを返す。
        """
        return self.data[index], self.labels[index]

    def __len__(self):
        """csv の行数を返す。
        """
        return len(self.data)

# Dataset を作成する。

batch_size = 64
dataset = Mydateset("NCI1_dataset_oversample.csv")
# dataloader = DataLoader(dataset, batch_size=batch_size)
train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,valid_size, test_size],generator=torch.Generator().manual_seed(1))
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset,batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


num = 100
lr = 0.005
criterion = torch.nn.CrossEntropyLoss()
model = MyModel()

patience = 5
early_stopping = EarlyStopping(patience, verbose=True)

#
# optimizer = torch.optim.Adam(model.parameters(),lr, weight_decay=0)
optimizer = torch.optim.SGD(model.parameters(),lr)
# device=torch.device("cuda"if torch.cuda.is_available() else "cpu")
scheduler = ReduceLROnPlateau(optimizer, mode='min',patience = 2,factor=0.1)

#
for e in tqdm(range(num)):
    print('Epoch {}/{}'.format(e + 1, num))
    print('-------------')
    model.train()
    epoch_loss = []

    train_total = 0
    train_correct = 0

    train_preds = []
    train_trues = []
    #train
    for x, y in tqdm(train_dataloader):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.data)

        _, predict = torch.max(out.data, 1)
        train_total += y.shape[0] * 1.0
        train_correct += int((y == predict).sum())

        #评估指标
        train_outputs = out.argmax(dim=1)

        train_preds.extend(train_outputs.detach().cpu().numpy())
        train_trues.extend(y.detach().cpu().numpy())

    epoch_loss = np.average(epoch_loss)
    scheduler.step(epoch_loss)

    early_stopping(epoch_loss, model)
    # 若满足 early stopping 要求
    if early_stopping.early_stop:
        print("Early stopping")
        # 结束模型训练
        break

    #f1_score
    sklearn_f1 = f1_score(train_trues, train_preds)
    print('train Loss: {:.4f} Acc:{:.4f} f1:{:.4f}'.format(epoch_loss,train_correct/train_total,sklearn_f1))

    print(confusion_matrix(train_trues, train_preds))

    #----------------------------------------------------------valid------------------------------------------------------------
    correct = 0
    total = 0

    valid_preds = []
    valid_trues = []

    with torch.no_grad():
        model.eval()
        for i, (x, labels) in enumerate(valid_dataloader):
            outputs = model(x)
            _, predict = torch.max(outputs.data, 1)
            total += labels.shape[0] * 1.0
            correct += int((labels == predict).sum())

            valid_outputs = outputs.argmax(dim=1)
            valid_preds.extend(valid_outputs.detach().cpu().numpy())
            valid_trues.extend(labels.detach().cpu().numpy())

        sklearn_f1 = f1_score(valid_trues, valid_preds)

        print('val Acc: {:.4f} f1:{:.4f}'.format(correct / total,sklearn_f1))

        print(confusion_matrix(valid_trues, valid_preds))

torch.save(model.state_dict(), 'model.pth')

#test
correct = 0
total = 0
test_preds = []
test_trues = []
with torch.no_grad():
    model.eval()
    for i, (x, labels) in enumerate(test_dataloader):
        outputs = model(x)
        _, predict = torch.max(outputs.data, 1)
        total += labels.shape[0] * 1.0
        correct += int((labels == predict).sum())

        test_outputs = outputs.argmax(dim=1)
        test_preds.extend(test_outputs.detach().cpu().numpy())
        test_trues.extend(labels.detach().cpu().numpy())

    sklearn_f1 = f1_score(test_trues, test_preds)

    print('test Acc: {:.4f} f1:{:.4f}'.format(correct/total,sklearn_f1))










