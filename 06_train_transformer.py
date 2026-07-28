#!/usr/bin/env python
# coding: utf-8
"""
Training script for a Transformer-based weather classification model.

Usage:
    python 06_train_transformer.py <s|w> <start> <end> [data_dir]

Arguments:
    s        Summer season
    w        Winter season
    start    Start repetition index
    end      End repetition index (exclusive)
    data_dir Optional path to the base data directory.
             Defaults to the script's own directory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys, time
from torch.utils.data import DataLoader, Dataset
from skimage import transform
from sklearn import preprocessing
import os
import numpy as np
import xarray as xr
import random
from math import sqrt
import config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print ('Number of arguments:', len(sys.argv), 'arguments.')
print ('Argument list:', str(sys.argv))

_base_data_dir = sys.argv[4] if len(sys.argv) > 4 else config.BASE_DIR

if sys.argv[1] == "s":
    prepath = os.path.join(_base_data_dir, "results", "summer") + "/"
    z500file = os.path.join(_base_data_dir, "data", "labeled", "summer-z500-labled-c5-7.nc")
    t2mfile = os.path.join(_base_data_dir, "data", "labeled", "summer-t2m-t2man-labled-c5-7.nc")
elif sys.argv[1] == "w":
    prepath = os.path.join(_base_data_dir, "results", "winter") + "/"
    z500file = os.path.join(_base_data_dir, "data", "labeled", "winter-z500-labled-c5-7.nc")
    t2mfile = os.path.join(_base_data_dir, "data", "labeled", "winter-t2m-t2man-labled-c5-7.nc")

ts = int(sys.argv[2])
te = int(sys.argv[3])

z500ds  =  xr.open_dataset(z500file)
t2mds  =  xr.open_dataset(t2mfile)

day = 1
p = 4
num_classes = len(z500ds.classnum)
withT2m = False
normal = True
input_dim = 1
input_w  =  50
input_h  =  80
net_epochs = 100
batch_size = 32

# Data preprocessing
# Generate index sets per class
alllabels =[]
lv = z500ds.labels.values

# Group by class
def builddata():
    global alllabels
    alllabels =[]
    for i in  range(0,num_classes):
        tl = np.argwhere(lv == i).squeeze()
        random.shuffle(tl)
        alllabels.append(tl)

builddata()

# Dataset
class MyDataset(Dataset):
    def __init__(self,day=1,p=1,withT2m = False ,normal = True , mode='train'):
        global input_dim

        self.z500 = z500ds.z500.values
        self.t2man = t2mds.t2man.values
        z500shape = self.z500.shape
        t2manshape =self.t2man.shape

        self.normal = normal
        # Normalization
        if normal :
            min_max_scaler = preprocessing.MinMaxScaler()
            self.z500 = min_max_scaler.fit_transform(self.z500.reshape(-1,z500shape[2]*z500shape[3])).reshape(z500shape)

            self.t2man = min_max_scaler.fit_transform(self.t2man.reshape(-1,t2manshape[2]*t2manshape[3])).reshape(t2manshape)

        self.z500 = transform.resize(self.z500, (self.z500.shape[0],self.z500.shape[1],input_h,input_w))
        self.t2man = transform.resize(self.t2man, (self.t2man.shape[0],self.t2man.shape[1],input_h,input_w))

        self.labels = z500ds.labels.values

        self.day = day
        self.p = p
        self.mode = mode

        self.withT2m = withT2m
        if self.withT2m :
            self.data = np.concatenate((self.z500[:,np.newaxis,self.day,:],self.t2man[:,np.newaxis,self.day,:]),axis = 1)
            input_dim = 2
        else:
            input_dim = 1
            self.data = self.z500[:,np.newaxis,self.day,:]

        print(len(self.labels))
        for i in  range(0,num_classes):
            print(i,len(self.labels[self.labels==i]))

        dlent = []
        dlene = []
        # Sample by class with ratio
        for i in  range(0,num_classes):
            tl = alllabels[i]
            tstart = int(len(tl) * 0.75)
            loc = int(tstart * p)
            tlt =  tl[0:loc]
            tle =  tl[tstart:]
            dlent.extend(tlt)
            dlene.extend(tle)

        if mode == 'train':
            self.data = self.data[dlent]
            self.labels = self.labels[dlent]
        else:
            self.data = self.data[dlene]
            self.labels = self.labels[dlene]

        print(len(self.labels))
        for i in  range(0,num_classes):
            print(i,len(self.labels[self.labels==i]))

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label

    def __len__(self):
        return len(self.labels)

    def print_sample(self, index: int = 0):
        indata,lable= self.__getitem__(index)
        print("sample", indata ,indata.shape, lable,lable.shape)
        return indata
    def _info(self,):
        info = "day-"+str(self.day)+"-p-"+str(self.p)
        if self.withT2m :
            info += "-withT2m"
        if self.normal :
            info +="-normal"
        info += "-"+self.mode+"-"
        return  info

class PositionalEncoding(nn.Module):
    """Relative positional encoding considering 2D image spatial relationships."""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        self.row_emb = nn.Embedding(max_len, d_model//2)
        self.col_emb = nn.Embedding(max_len, d_model//2)

    def forward(self, x, positions):
        row_emb = self.row_emb(positions[:, :, 0])
        col_emb = self.col_emb(positions[:, :, 1])
        pos_emb = torch.cat([row_emb, col_emb], dim=-1)
        return x + pos_emb

class MultiHeadAttention(nn.Module):
    """Multi-head attention with learnable scaling factor."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.scale = nn.Parameter(torch.ones(1) * sqrt(self.d_k))

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(out)

class FeedForward(nn.Module):
    """Feed-forward network with GELU activation and layer scaling."""
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.gelu = nn.GELU()
        self.gamma = nn.Parameter(torch.ones(d_model))
    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x))) * self.gamma

class TransformerBlock(nn.Module):
    """Transformer encoder block with residual connections and layer normalization."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model)
    def forward(self, x):
        attn_out = self.attention(x, x, x)
        x = x + attn_out
        ffn_out = self.ffn(x)
        x = x + ffn_out
        return x

class WeatherTransformer(nn.Module):
    """Weather classification Transformer model."""
    def __init__(self, num_classes=5, img_size=(input_h,input_w), patch_size=(10,10),
                 d_model=256, num_heads=8, num_layers=2):
        super().__init__()
        global input_dim
        h, w = img_size
        ph, pw = patch_size
        assert h % ph == 0 and w % pw == 0, "Image size must be divisible by patch size"

        self.num_patches = (h//ph) * (w//pw)
        self.patch_emb = nn.Conv2d(input_dim, d_model, kernel_size=(ph,pw), stride=(ph,pw))
        rows = torch.arange(0, h//ph).repeat_interleave(w//pw)
        cols = torch.arange(0, w//pw).repeat(h//ph)
        self.register_buffer('positions', torch.stack([rows, cols], dim=1))

        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Sequential(*[
            TransformerBlock(d_model, num_heads) for _ in range(num_layers)
        ])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.patch_emb(x)
        x = x.flatten(2).transpose(1, 2)

        positions = self.positions.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.pos_encoder(x, positions)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        x = self.transformer(x)

        return self.classifier(x[:, 0, :])

class resultdata():
    def __init__(self,name):
        self.name = name
        self.data = np.zeros((num_classes+2), dtype=float)
        self.num_classes = num_classes
        self.day = day
        self.p = p
    def printme(self):
        print(self.name ,"day:",self.day,"p:",self.p)
        print("accuray: ",self.data[self.num_classes])
        print("all recall:",self.data[self.num_classes+1])
        for i in range(0,self.num_classes):
            print("class"+str(i)+" recall:",self.data[i])

def save_model(model,path):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(model.state_dict(),path)

def load_model(model,path):
    model.load_state_dict(torch.load(path), strict=False)
    print('######## MODEL LOADED ########')
    return model

best_recall = 0.0
best_acc = 0.0

def train(model,modelname,lr = 0.001,endacc = 1,endloss = 0,saverecall= -1):
    global net_epochs, best_recall,train_dataset,train_loader,test_dataset,test_loader,input_dim,best_acc,num_classes
    best_recall = 0.0
    best_acc = 0.0
    print(best_recall)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:\t',device)

    print('==>>> total training batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}\n'.format(len(test_loader)))

    model = model.to(device)
    print(model)
    print ("# parameters: ", sum(param.numel() for param in model.parameters()))

    loss_func=torch.nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(),lr=lr)

    for epoch in range(net_epochs):
        # training
        avg_loss = 0.0
        train_acc=0.0
        all_acc = 0.0
        all_label  = 0.0
        total_acc = 0.0
        start_time = time.time()
        for batch_no, (x, target) in enumerate(train_loader):

            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = loss_func(out,target)
            loss.backward()
            optimizer.step()

            _, pred_label = torch.max(out, dim=1)
            pred_label = pred_label.to(device)
            train_acc = (pred_label == target.data).double().sum()
            all_acc += train_acc
            all_label += len(pred_label)
            if batch_no%10==0:
                sys.stdout.write('Epoch = {0}\t Batch n.o.={1}\t Loss={2:.4f}\t Batch_acc={3:.4f}\r'.format(epoch,batch_no,loss.item(),train_acc/len(pred_label)))
                sys.stdout.flush()
            avg_loss+=loss.item()
        total_time = time.time()-start_time
        total_acc = all_acc/all_label
        print('\nAvg Loss={0:.2f}\t Accuracy={1:.2f}\t accnum={2:.0f}\t allnum={3:.0f}\t time taken = {4:0.2f}'.format(avg_loss/len(train_loader),total_acc,all_acc,all_label,total_time))
        if total_acc > endacc and avg_loss < endloss:
            saverecall = 0
            break
        if saverecall>0:
            saverecall = test(model,modelname,saverecall,epoch)
    if saverecall<=0:
        test(model,modelname,saverecall,epoch)

def test(model,modelname,saverecall=-1,epoch=-1):
        # Testing
        result = []
        yl = []
        correct_cnt=0
        total_cnt = 0
        for batch_idx, (x, target) in enumerate(test_loader):
            x, target = x.to(device), target.to(device)
            out = model(x)

            _, pred_label = torch.max(out, dim=1)

            result.extend(pred_label.cpu().numpy())
            yl.extend(target.cpu().numpy())

            pred_label=pred_label.to(device)
            total_cnt += x.data.size()[0]
            correct_cnt += (pred_label == target).sum()
        test_acc = correct_cnt.item()*1.0/total_cnt
        print('Test Accuracy={}'.format(test_acc))
        # Prediction / validation
        Rdata =  resultdata(modelname)
        yl = np.asarray(yl)
        result = np.asarray(result)
        ans = len(yl[yl==0])
        ac = yl==result
        test_acc2 = len(ac[ac])/len(ac)
        Rdata.data[num_classes] = test_acc2
        z = yl[np.where(yl!=0)] == result[np.where(yl!=0)]
        test_recall= len(z[z])/len(z)
        r = result[:]
        y = yl[:]
        Rdata.data[num_classes+1] = test_recall
        for i in range(0,num_classes):
            zt = r[np.where(y==i)]==i
            Rdata.data[i] = len(zt[zt])/len(zt)
        Rdata.printme()
        if test_recall>saverecall:
            save_model(model,prepath+'{}/{}_{}_{}_recall_{:.3f}_acc_{:.3f}_epoch_{}_er_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}.pkl'
                       .format(modelname,modelname,input_dim,test_dataset._info(),test_recall,test_acc2,epoch,saverecall,Rdata.data[0],Rdata.data[1],Rdata.data[2],Rdata.data[3],Rdata.data[4],Rdata.data[5]))
            return test_recall
        else:
            return saverecall

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
best_recall = 0.0
mymodel = WeatherTransformer
mymodelname = "transformer"

for x in range(1,3):
    withT2m = x!=1
    input_dim = x
    for i in [1,0.75,0.5,0.25]:
        p = i
        for j in [1,2,3,4,5]:
            day = j

            # Training data loading
            train_dataset = MyDataset(day,p,withT2m,normal,mode='train')
            train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)

            # Test data loading
            test_dataset = MyDataset(day,p,withT2m, normal,mode='eval')
            test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)

            net_epochs = 100

            # Define model parameters
            input_shape = (input_h,input_w)
            for n in range(ts,te):

                save_path = prepath + mymodelname + "-" +str(n)

                # Initialize model
                model = mymodel()

                train(model,mymodelname+"-"+str(n),0.0001,0.9999,0.0001,0.2)