import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
#from torchvision import transforms
#from PIL import Image
import torch.distributed as dist


import os
import subprocess
from mpi4py import MPI

cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor



#im = Image.open("name.jpg")
#pix = im.load()

import h5py
import time

# Data Augmentation
train_transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                 transforms.RandomVerticalFlip(),
                 transforms.RandomCrop(size=[32,32], padding=2),
                 transforms.ColorJitter(brightness=0, contrast = 0),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.507,0.487,0.441], std=[0.267, 0.256, 0.276])])
test_transform = transforms.Compose(
                 [transforms.ToTensor(),
                 transforms.Normalize(mean=[0.507,0.487,0.441], std=[0.267,0.256,0.276])])

# For trainning data
train_dataset = torchvision.datasets.CIFAR100(root=’~/scratch/’,
                                        train=True,download=False, transform=train_transform)
train_loader = torch.utils.data.DataLoader(trainset,
                                        batch_size=batch_size, shuffle=True, num_workers=0)
# For testing data
test_dataset = torchvision.datasets.CIFAR100(root=’~/scratch/’,
                                        train=False,download=False, transform=test_transform)
test_loader = torch.utils.data.DataLoader(testset,
                                        batch_size=batch_size, shuffle=False, num_workers=0)



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


#num_blocks = [2,4,4,2]
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100, zero_init_residual=False,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))


        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.conv2 = self._make_layer(block, 32, layers[0])
        self.conv3 = self._make_layer(block, 64, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.conv4 = self._make_layer(block, 128, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.conv5 = self._make_layer(block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.pool = nn.MaxPool2d(2,2)
        self.fc = nn.Linear(1024 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    
    
    
model = ResNet(BasicBlock, [2,4,4,2], num_classes=num_classes)

#Make sure that all nodes have the same model
for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
    param.data = tensor0/np.sqrt(np.float(num_nodes))


model.cuda()

LR = 0.001
batch_size = 100
num_epochs = 250

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LR)



for epoch in range(num_epochs):  
    start_time = time.time()
    model.train()
    for images, labels in train_loader:
        # Move tensors to GPU
        images = images.cuda()
        labels = labels.cuda()
        # Forward Pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
        # Backwards pass

        loss.backward()    

        for param in model.parameters():
            tensor0 = param.grad.data.cpu()
            dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
            tensor0 /= float(num_nodes)
            param.grad.data = tensor0.cuda()        

        optimizer.step()   
        
    #Test Loss
    model.eval()
    correct = 0
    total = 0 
    for images, labels in test_loader:
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    test_accuracy = correct/total
    print('Epoch {}, Time {:.4f}, Loss: {:.4f}, Train Accuracy: {:.4f}, Test Accuracy: {:.4f}'.format(epoch,time.time() - start_time, loss.item(), train_accuracy, test_accuracy))
    torch.save(model.state_dict(), 'epoch-{}.ckpt'.format(epoch))
    