import torch
import torch.nn as nn
from torchvision.models import resnet18
import os

import random as rd
import cv2
import numpy as np
from progressbar import *
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
PATH = 'data'
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional


class BCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(BCNN, self).__init__()
        features = torchvision.models.resnet34(pretrained=pretrained)
        # Remove the pooling layer and full connection layer
        self.conv = nn.Sequential(*list(features.children())[:-2])
        self.fc = nn.Linear(24, num_classes)
        self.softmax = nn.Softmax(dim=1)

        if pretrained:
            for parameter in self.conv.parameters():
                parameter.requires_grad = False
            nn.init.kaiming_normal_(self.fc.weight.data)
            nn.init.constant_(self.fc.bias, val=0)

    def forward(self, input):
        features = self.conv(input)
        # Cross product operation
        features = features.view(features.size(0), -1)
        features_T = torch.transpose(features, 0, 1)
        features = torch.matmul(features, features_T) / (14 * 14)

        features = features.view(features.size(0), -1)
        # The signed square root
        features = torch.sign(features) * torch.sqrt(torch.abs(features) + 1e-12)
        # L2 regularization
        features = torch.nn.functional.normalize(features)
        out = self.fc(features)
        softmax = self.softmax(out)
        # print(softmax.shape)
        return out, softmax


class Classification(object):
    def __init__(self, batch_size, epoch):
        self.batch_size = batch_size
        self.epoch = epoch
        self.model = BCNN(2, pretrained=False)

    def _load_data(self):
        self.train_imgs, self.train_labels = [], []
        for filename in os.listdir(os.path.join(os.getcwd(), 'data', 'train', 'bee')):
            img = cv2.imread(os.path.join(os.getcwd(), 'data', 'train', 'bee', filename))
            if img is not None:
                img = cv2.resize(img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
                self.train_labels.append(1)
                self.train_imgs.append(img)
        for filename in os.listdir(os.path.join(os.getcwd(), 'data', 'train', 'other')):
            img = cv2.imread(os.path.join(os.getcwd(), 'data', 'train', 'other', filename))
            if img is not None:
                img = cv2.resize(img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
                self.train_labels.append(0)
                self.train_imgs.append(img)

        self.test_imgs, self.test_labels = [], []
        for filename in os.listdir(os.path.join(os.getcwd(), 'data', 'test', 'bee')):
            img = cv2.imread(os.path.join(os.getcwd(), 'data', 'test', 'bee', filename))
            if img is not None:
                img = cv2.resize(img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
                self.test_labels.append(0)
                self.test_imgs.append(img)
        for filename in os.listdir(os.path.join(os.getcwd(), 'data', 'test', 'other')):
            img = cv2.imread(os.path.join(os.getcwd(), 'data', 'test', 'other', filename))
            if img is not None:
                img = cv2.resize(img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
                self.test_labels.append(0)
                self.test_imgs.append(img)
        return

    def _load_dataset(self):
        X_train_tensor = torch.Tensor(np.array(self.train_imgs) / 255)  # transform to torch tensor
        y_train_tensor = torch.Tensor(self.train_labels).long()
        X_test_tensor = torch.Tensor(np.array(self.test_imgs) / 255)
        y_test_tensor = torch.Tensor(self.test_labels).long()

        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)  # create datset
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        # create data loader
        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        return

    def train(self, train_loader, criterion, optimizer, epoch):
        train_loss = 0
        correct = 0
        self.model.train()
        for batch_idx, item in enumerate(train_loader):
            try:
                data, target = item
                data = Variable(data.reshape(-1, 3, 84, 84))
                target = Variable(target)
                optimizer.zero_grad()
                output, _ = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                if batch_idx == len(train_loader) - 1:
                    print('Epoch {}, Training Loss: {:.4f}'.format(epoch, train_loss / (batch_idx + 1)))
            except:
                continue
            acc = correct / len(train_loader.dataset)
        return (acc)

    def test(self, test_loader, criterion, epoch):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = Variable(data.reshape(-1, 3, 84, 84))
                target = Variable(target)
                output, _ = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset)
        # test_loss = (test_loss*batch_size)/len(test_loader.dataset)
        print('Test{}: Accuracy: {:.4f}%'.format(epoch, 100. * acc))
        return (acc)

    def run(self):
        self._load_data()
        self._load_dataset()
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([10, 1]))
        # optimizer = optim.SGD(self.model.parameters(),lr=0.001,momentum=0.9)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        train_acc = []
        test_acc = []
        old = 0
        for epoch in range(1, self.epoch + 1):
            train_acc.append(self.train(self.train_loader, criterion, optimizer, epoch))
            acc = self.test(self.test_loader, criterion, epoch)
            test_acc.append(acc)
            if epoch == 1 or old < acc:
                torch.save(self.model.state_dict(), 'ckpt_blcnn_1.pth')
                old = acc
        return

def get_probability(model, img):
    _, pro = model(img)
    probability = pro.detach().numpy()[0][1]
    return probability


if __name__ == '__main__':
    classification = Classification(24, 10)
    classification.run()

    model = BCNN(2, pretrained=False)
    model.load_state_dict(torch.load('ckpt_blcnn_1.pth'))
    model.eval()
    # print(model.state_dict())

    res = []
    for imgpath in os.listdir('data/unverified'):
        try:
            img = cv2.imread(os.path.join('data/unverified', imgpath))
            if img is not None:
                img = cv2.resize(img, dsize=(84, 84), interpolation=cv2.INTER_CUBIC)
            img_tensor = np.array([img]*24)
            # print(img_tensor.shape)
            img = torch.Tensor(np.array(img_tensor) / 255)
            img = img.reshape(-1, 3, 84, 84)
            # print(img.shape)
            pro = get_probability(model, img)
            print(pro)
            res.append((imgpath, pro))
        except:
            continue

    with open('unverified_pro.txt', 'a') as fout:
        for r in res:
            fout.write(r[0]+' '+str(r[1])+'\n')
