
import os
import json

import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm

from model_alexNet import AlexNet

# from model_resneXt import resnet34
from model_resneXt import resnext50_32x4d



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_set = datasets.CIFAR100('../data', train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR100('../data', train=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(
        train_set,batch_size=64,shuffle=True,num_workers=1)
    test_loader = torch.utils.data.DataLoader(
        test_set,batch_size=64,shuffle=False,num_workers=1)
    tran_num = train_set.__len__()
    val_num = test_set.__len__()
    # print(tran_num)
    # print(val_num)

    # net = AlexNet(num_classes=10, init_weights=True)
    # net = AlexNet(num_classes=100) # todo AlexNet
    # save_path = './checkpoint/AlexNet.pth'

    # net = resnet34(num_classes=100) #todo resnet34
    # save_path = './checkpoint/resnet34.pth'

    net = resnext50_32x4d(num_classes=100)  # todo resnet34
    save_path = './checkpoint/resnext50_32x4d.pth'

    net.to(device)


    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    epochs = 2

    best_acc = 0.0
    train_steps = len(train_loader)

    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images.to(device))
            loss = loss_function(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            # acc1, acc5 = accuracy(outputs, labels, topk=(1, 5))
            # losses.update(loss.item(), data.size(0))
            # top1.update(acc1[0], data.size(0))
            # top5.update(acc5[0], data.size(0))

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(test_loader)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
                # acc += accuracy(predict_y,val_labels)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')



if __name__ =="__main__":
    main()
