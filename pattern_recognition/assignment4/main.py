import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tensorboardX import SummaryWriter
import os


class MyMNIST(Dataset):
    train_file = 'TrainSamples.csv'
    label_file = 'TrainLabels.csv'

    def __init__(self, k=1, val=False, trainval=False):
        data = np.loadtxt("TrainSamples.csv", dtype=np.float, delimiter=",")
        label = np.loadtxt("TrainLabels.csv", dtype=np.float, delimiter=",")
        val_start_index = 2000 * (k - 1)
        val_end_index = 2000 * k
        if not trainval:
            if val:
                self.data = data[val_start_index : val_end_index]
                self.label = label[val_start_index : val_end_index]
            else:
                self.data = np.concatenate((data[: val_start_index], data[val_end_index :]))
                self.label = np.concatenate((label[: val_start_index], label[val_end_index :]))
        else:
            self.data = data 
            self.label = label
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # return self.data[index].reshape((1, 9, 9)), self.label[index]
        return self.data[index], self.label[index]


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)

    _, pred = output.topk(1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = correct.view(-1).float().sum(0, keepdim=True).mul_(100.0 /batch_size)
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, length=0):
        self.length = length
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count



     
class SimpleNet(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(in_dim, n_hidden_1)
        self.layer1_2 = nn.Linear(n_hidden_1, n_hidden_1)
        self.layer2 = nn.Linear(n_hidden_1, n_hidden_2)
        self.layer3 = nn.Linear(n_hidden_2, out_dim)

    def forward(self, x):
        x = F.sigmoid(self.layer1(x))
        x = F.dropout(x, training=self.training)
        x = F.sigmoid(self.layer1_2(x))
        x = F.dropout(x, training=self.training)

        x = F.sigmoid(self.layer2(x))
        x = F.dropout(x, training=self.training)
        x = self.layer3(x)
        return x



def train(train_loader, model, criterion, optimizer):
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    model.train()
    for i, (input_data, target) in enumerate(train_loader):
        input_var = \
        torch.autograd.Variable(input_data.type(torch.FloatTensor)).cpu()
        target_var = \
        torch.autograd.Variable(target.type(torch.LongTensor)).cpu()

        output = model(input_var)
        
        loss = criterion(output, target_var)

        predict = F.softmax(output, dim=1)
        acc = accuracy(predict, target_var)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        loss_.update(loss.data)
        acc_.update(acc.data)
    
    return loss_.avg, acc_.avg


def validate(val_loader, model, criterion):
    loss_ = AverageMeter()
    acc_ = AverageMeter()
    model.eval()
    for i, (input_data, target) in enumerate(val_loader):
        input_var = \
        torch.autograd.Variable(input_data.type(torch.FloatTensor)).cpu()
        target_var = \
        torch.autograd.Variable(target.type(torch.LongTensor)).cpu()

        output = model(input_var)
        predict = F.softmax(output, dim=1)

        loss = criterion(output, target_var)
        acc = accuracy(predict, target_var)

        loss_.update(loss)
        acc_.update(acc)

    return loss_.avg, acc_.avg

def main():
	base_lr = 5e-3
	weight_decay = 1e-5
	n_epoch = 2000

	print('start')
	for k in range(2, 3):
		writer = SummaryWriter()
		
		train_data = MyMNIST(k, trainval=True)
		train_loader = DataLoader(dataset=train_data, shuffle=True, num_workers=2, batch_size=20000)
		val_data = MyMNIST(k, val=True)
		val_loader = DataLoader(dataset=val_data, shuffle=True, num_workers=2, batch_size=2000)

		model = SimpleNet(84, 500, 200, 10)

		model = model.cpu()
		criterion = nn.CrossEntropyLoss()
		optimizer = torch.optim.Adam(params=model.parameters(), lr=base_lr, weight_decay=weight_decay)

		for epoch in range(n_epoch):
			current_lr = optimizer.param_groups[0]['lr']

			train_loss, train_acc = train(train_loader, model, criterion, optimizer)
			eval_loss, eval_acc = validate(val_loader, model, criterion)

			writer.add_scalar('train_loss', train_loss, epoch)
			writer.add_scalar('train_acc', train_acc, epoch)
			writer.add_scalar('eval_loss', eval_loss, epoch)
			writer.add_scalar('eval_acc', eval_acc, epoch)
			writer.add_scalar('current_lr', current_lr, epoch)
			print('epoch:{}'.format(epoch)
			print('train_loss:{}'.format(train_loss))

		torch.save(model.state_dict(), 'final.pth')
	print("end")

if __name__ == "__main__":
	main()
