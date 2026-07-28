#!/usr/bin/env python
# coding: utf-8

"""
Capsule Network training script.

Usage: python 05_train_capsule.py s 0 24 (s=summer / w=winter, start end)
"""

import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import xarray as xr
from torch.utils.data import DataLoader, Dataset
from skimage import transform
from sklearn import preprocessing
import config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument list:', str(sys.argv))

if sys.argv[1] == "s":
    prepath = config.RESULTS_SUMMER_DIR + "/"
    z500file = os.path.join(config.LABELED_DATA_DIR, "summer-z500-labled-c5-7.nc")
    t2mfile = os.path.join(config.LABELED_DATA_DIR, "summer-t2m-t2man-labled-c5-7.nc")
elif sys.argv[1] == "w":
    prepath = config.RESULTS_WINTER_DIR + "/"
    z500file = os.path.join(config.LABELED_DATA_DIR, "winter-z500-labled-c5-7.nc")
    t2mfile = os.path.join(config.LABELED_DATA_DIR, "winter-t2m-t2man-labled-c5-7.nc")

ts = int(sys.argv[2])
te = int(sys.argv[3])

z500ds = xr.open_dataset(z500file)
t2mds = xr.open_dataset(t2mfile)

day = 1
p = 1
num_classes = len(z500ds.classnum)
withT2m = False
normal = True
input_dim = 1
input_w = 28
input_h = 28
net_epochs = 50
batch_size = 32

alllabels = []
lv = z500ds.labels.values

def builddata():
    global alllabels
    alllabels = []
    for i in range(0, num_classes):
        tl = np.argwhere(lv == i).squeeze()
        random.shuffle(tl)
        alllabels.append(tl)

builddata()

class MyDataset(Dataset):
    def __init__(self, day=1, p=1, withT2m=False, normal=True, mode='train'):
        global input_dim

        self.z500 = z500ds.z500.values
        self.t2man = t2mds.t2man.values
        z500shape = self.z500.shape
        t2manshape = self.t2man.shape

        self.normal = normal
        if normal:
            min_max_scaler = preprocessing.MinMaxScaler()
            self.z500 = min_max_scaler.fit_transform(self.z500.reshape(-1, z500shape[2] * z500shape[3])).reshape(z500shape)

            self.t2man = min_max_scaler.fit_transform(self.t2man.reshape(-1, t2manshape[2] * t2manshape[3])).reshape(t2manshape)

        self.z500 = transform.resize(self.z500, (self.z500.shape[0], self.z500.shape[1], input_w, input_h))
        self.t2man = transform.resize(self.t2man, (self.t2man.shape[0], self.t2man.shape[1], input_w, input_h))

        self.labels = z500ds.labels.values

        self.day = day
        self.p = p
        self.mode = mode

        self.withT2m = withT2m
        if self.withT2m:
            self.data = np.concatenate((self.z500[:, np.newaxis, self.day, :], self.t2man[:, np.newaxis, self.day, :]), axis=1)
            input_dim = 2
        else:
            input_dim = 1
            self.data = self.z500[:, np.newaxis, self.day, :]

        print(len(self.labels))
        for i in range(0, num_classes):
            print(i, len(self.labels[self.labels == i]))

        dlent = []
        dlene = []
        for i in range(0, num_classes):
            tl = alllabels[i]
            tstart = int(len(tl) * 0.75)
            loc = int(tstart * p)
            tlt = tl[0:loc]
            tle = tl[tstart:]
            dlent.extend(tlt)
            dlene.extend(tle)

        if mode == 'train':
            self.data = self.data[dlent]
            self.labels = self.labels[dlent]
        else:
            self.data = self.data[dlene]
            self.labels = self.labels[dlene]

        print(len(self.labels))
        for i in range(0, num_classes):
            print(i, len(self.labels[self.labels == i]))

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label

    def __len__(self):
        return len(self.labels)

    def print_sample(self, index: int = 0):
        indata, lable = self.__getitem__(index)
        print("sample", indata, indata.shape, lable, lable.shape)
        return indata

    def _info(self, ):
        info = "day-" + str(self.day) + "-p-" + str(self.p)
        if self.withT2m:
            info += "-withT2m"
        if self.normal:
            info += "-normal"
        info += "-" + self.mode + "-"
        return info


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Squash(x):
    l2norm = x.norm(dim=-1, keepdim=True)
    unit_v = x / l2norm
    squashed_v = l2norm.pow(2) / (1 + l2norm.pow(2))
    x = unit_v * squashed_v
    return x

class Capsule_conv(nn.Module):
    def __init__(self, in_channels, out_channels, cap_dim):
        super(Capsule_conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cap_dim = cap_dim
        self.kernel_size = 9
        self.stride = 2
        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=(self.out_channels * self.cap_dim),
                              kernel_size=self.kernel_size, stride=self.stride)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1, self.cap_dim)
        x = Squash(x)
        return x

class Capsule_fc(nn.Module):
    def __init__(self, in_cap_dim, num_in_caps, out_cap_dim, num_out_caps, r=3):
        super(Capsule_fc, self).__init__()
        self.num_in_caps = num_in_caps
        self.num_out_caps = num_out_caps
        self.in_cap_dim = in_cap_dim
        self.out_cap_dim = out_cap_dim
        self.W = nn.Parameter(torch.randn(self.num_in_caps, self.num_out_caps, self.out_cap_dim, self.in_cap_dim))
        self.routing_iterations = r

    def forward(self, x):
        x = torch.matmul(self.W, x.unsqueeze(-1).unsqueeze(-3)).squeeze(4)
        coupling_coef = torch.zeros([*x.shape[:-1]]).unsqueeze(-1)
        coupling_coef.requires_grad_()
        coupling_coef = coupling_coef.to(device)
        b = coupling_coef
        for r in range(1, self.routing_iterations + 1):
            coupling_coef = F.softmax(b, dim=1)
            s = coupling_coef * x
            s = s.sum(dim=1, keepdim=True)
            v = Squash(s)
            if r != self.routing_iterations:
                b = b + (v * x).sum(dim=-1, keepdim=True)
        return v.squeeze(1)

def MarginLoss(output, one_hot):
    downweighting = 0.5
    m_plus = 0.8
    m_minus = 0.2
    l2norm = output.norm(dim=-1)
    term1 = F.relu(m_plus - l2norm) ** 2
    term2 = F.relu(l2norm - m_minus) ** 2
    loss_vec = one_hot * term1 + downweighting * ((1 - one_hot) * term2)
    total_loss = loss_vec.sum(dim=-1)
    return total_loss.mean()

def ReconLoss(original, recon):
    original = original.view(-1, input_w * input_h)
    recon = recon.view(-1, input_w * input_h)
    loss_vec = (recon - original) ** 2
    loss_vec = loss_vec.sum(-1)
    return loss_vec.mean()

def CapsuleLoss(out, label, original, recon):
    loss_m = MarginLoss(out, label)
    loss_r = ReconLoss(original, recon)
    loss = loss_m + 0.0005 * loss_r
    return loss

class Capsule_Net(nn.Module):
    def __init__(self):
        super(Capsule_Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_dim,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=0
            ),
            nn.ReLU(),
        )
        self.primary_caps = Capsule_conv(64, 8, 8)
        self.digcaps = Capsule_fc(8, 8 * 6 * 6, 16, num_classes)
        self.decoder = nn.Sequential(
            nn.Linear(num_classes * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784 * input_dim),
            nn.Sigmoid()
        )
        self.mask = torch.eye(num_classes)
        self.mask.requires_grad_()
        self.mask = self.mask.to(device)

    def forward(self, x, label=None):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.primary_caps(x)
        x = self.digcaps(x)
        if label is None:
            logits = x.norm(dim=-1)
            _, label = torch.max(logits.data, dim=1)
            label = label.to(device)

        one_hot = self.mask.index_select(dim=0, index=label)
        recon = one_hot.unsqueeze(-1) * x
        recon = recon.view(-1, x.shape[1] * x.shape[2])
        recon = self.decoder(recon)
        return (x, recon, one_hot)


class resultdata():
    def __init__(self, name):
        self.name = name
        self.data = np.zeros((num_classes + 2), dtype=float)
        self.num_classes = num_classes
        self.day = day
        self.p = p

    def printme(self):
        print(self.name, "day:", self.day, "p:", self.p)
        print("accuray: ", self.data[self.num_classes])
        print("all recall:", self.data[self.num_classes + 1])
        for i in range(0, self.num_classes):
            print("class" + str(i) + " recall:", self.data[i])


def save_model(model, path):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path), strict=False)
    print('######## MODEL LOADED ########')
    return model


best_recall = 0.0
best_acc = 0.0

def train(model, modelname, lr=0.001, endacc=1, endloss=0, saverecall=-1):
    global best_recall, train_dataset, train_loader, test_dataset, test_loader, input_dim, num_classes, p, day
    print(best_recall)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:\t', device)

    print('==>>> total training batch number: {}'.format(len(train_loader)))
    print('==>>> total testing batch number: {}\n'.format(len(test_loader)))

    model = model.to(device)
    print(model)
    print("# parameters: ", sum(param.numel() for param in model.parameters()))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(net_epochs):
        avg_loss = 0.0
        train_acc = 0.0
        all_acc = 0.0
        all_label = 0.0
        total_acc = 0.0
        start_time = time.time()
        for batch_no, (x, target) in enumerate(train_loader):

            x, target = x.to(device), target.to(device)

            optimizer.zero_grad()
            out, recon, mask = model(x, target)
            loss = CapsuleLoss(out, mask, x, recon)
            loss.backward()
            optimizer.step()
            logits = F.softmax(out.norm(dim=-1), dim=-1)
            _, pred_label = torch.max(logits.data, dim=1)
            pred_label = pred_label.to(device)
            train_acc = (pred_label == target.data).double().sum()
            all_acc += train_acc
            all_label += len(pred_label)
            if batch_no % 10 == 0:
                sys.stdout.write('Epoch = {0}\t Batch n.o.={1}\t Loss={2:.4f}\t Batch_acc={3:.4f}\r'.format(epoch, batch_no, loss.item(), train_acc / batch_size))
                sys.stdout.flush()
            avg_loss += loss.item()
        total_time = time.time() - start_time
        total_acc = all_acc / all_label
        print('\nAvg Loss={0:.2f}\t Accuracy={1:.2f}\t accnum={2:.0f}\t allnum={3:.0f}\t time taken = {4:0.2f}'.format(avg_loss / len(train_loader), total_acc, all_acc, all_label, total_time))
        if total_acc > endacc and avg_loss < endloss:
            saverecall = 0
            break
        if saverecall > 0:
            saverecall = test(model, modelname, saverecall, epoch)

    if saverecall <= 0:
        test(model, modelname, saverecall, epoch)

def test(model, modelname, saverecall=-1, epoch=-1):
    result = []
    yl = []
    correct_cnt = 0
    total_cnt = 0
    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = x.to(device), target.to(device)
        out, recon, _ = model(x)
        logits = out.norm(dim=-1)
        _, pred_label = torch.max(logits.data, dim=1)

        result.extend(pred_label.cpu().numpy())
        yl.extend(target.cpu().numpy())

        pred_label = pred_label.to(device)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target).sum()
    test_acc = correct_cnt.item() * 1.0 / total_cnt
    print('Test Accuracy={}'.format(test_acc))
    Rdata = resultdata(modelname)
    yl = np.asarray(yl)
    result = np.asarray(result)
    ans = len(yl[yl == 0])
    ac = yl == result
    test_acc2 = len(ac[ac]) / len(ac)
    Rdata.data[num_classes] = test_acc2
    z = yl[np.where(yl != 0)] == result[np.where(yl != 0)]
    test_recall = len(z[z]) / len(z)
    r = result[:]
    y = yl[:]
    Rdata.data[num_classes + 1] = test_recall
    for i in range(0, num_classes):
        zt = r[np.where(y == i)] == i
        Rdata.data[i] = len(zt[zt]) / len(zt)
    Rdata.printme()
    if test_recall > saverecall:
        save_model(model, prepath + '{}/{}_{}_{}_recall_{:.3f}_acc_{:.3f}_epoch_{}_er_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}_{:.3f}.pkl'
                   .format(modelname, modelname, input_dim, test_dataset._info(), test_recall, test_acc2, epoch, saverecall, Rdata.data[0], Rdata.data[1], Rdata.data[2], Rdata.data[3], Rdata.data[4], Rdata.data[5]))
        return test_recall
    else:
        return saverecall


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mymodel = Capsule_Net
mymodelname = "Capsule"

for n in range(ts, te):
    save_path = prepath + mymodelname + "-" + str(n)
    if os.path.exists(save_path):
        print(save_path, "Directory exists, skipping training.")
        continue
    else:
        print(save_path, "Starting training.")
    builddata()
    for x in range(1, 3):
        withT2m = x != 1
        input_dim = x
        for i in [1, 0.75, 0.5, 0.25]:
            p = i
            for j in [1, 2, 3, 4, 5]:
                day = j
                train_dataset = MyDataset(day, p, withT2m, normal, mode='train')
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_dataset = MyDataset(day, p, withT2m, normal, mode='eval')
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

                net_epochs = 100
                train(mymodel(), mymodelname + "-" + str(n), 0.001, 0.999, 0.001, 0.2)