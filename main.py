"""
Created on Wed July 3 12:38:21 2024

@author: Feiyang Wu

Python 3.5.2
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from snn import *
import time
import os

names = 'spiking_model'
data_path =  './raw/' 
batch_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = torchvision.datasets.MNIST(root= data_path, train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_set = torchvision.datasets.MNIST(root= data_path, train=False, download=True,  transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
acc_record = list([])
loss_train_record = list([])
loss_test_record = list([])


net = SNN()
net.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    with torch.autograd.set_detect_anomaly(True):
        running_loss = 0
        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            net.zero_grad()
            optimizer.zero_grad()

            images = images.float().to(device)
            
            images = images.view(100,-1)

            outputs = net(images)
            labels_ = torch.zeros(batch_size, 10).scatter_(1, labels.view(-1, 1), 1)
            loss = criterion(outputs.cpu(), labels_)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            if (i+1)%100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.5f'
                        %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size,running_loss ))
                running_loss = 0
                print('Time elasped:', time.time()-start_time)
        correct = 0
        total = 0
        optimizer = lr_scheduler(optimizer, epoch, learning_rate, 40)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to(device)
                inputs = inputs.view(100,-1)
                optimizer.zero_grad()
                outputs = net(inputs)
                labels_ = torch.zeros(batch_size, 10).scatter_(1, targets.view(-1, 1), 1)
                loss = criterion(outputs.cpu(), labels_)
                _, predicted = outputs.cpu().max(1)
                total += float(targets.size(0))
                correct += float(predicted.eq(targets).sum().item())
                if batch_idx %100 ==0:
                    acc = 100. * float(correct) / float(total)
                    print(batch_idx, len(test_loader),' Acc: %.5f' % acc)

        print('Iters:', epoch,'\n\n\n')
        print('Test Accuracy of the model on the 10000 test images: %.3f' % (100 * correct / total))
        acc = 100. * float(correct) / float(total)
        acc_record.append(acc)
        if epoch % 5 == 0:
            print(acc)
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'acc_record': acc_record,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt' + names + '.t7')
            best_acc = acc