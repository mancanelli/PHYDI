import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ph_layers.hypercomplex_layers import PHConv

class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, n=4):
        super(FixupBasicBlock, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = PHConv(n, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bias1b = nn.Parameter(torch.zeros(1))
        #self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = PHConv(n, planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        #self.downsample = downsample
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                PHConv(n, in_planes, self.expansion*planes, kernel_size=1, stride=stride)
            )
            nn.init.normal_(self.shortcut[0].F, mean=0, std=np.sqrt(2 / (self.shortcut[0].F.shape[0] * np.prod(self.shortcut[0].F.shape[2:]))))
            nn.init.normal_(self.shortcut[0].A, mean=0, std=np.sqrt(2 / (self.shortcut[0].A.shape[0] * np.prod(self.shortcut[0].A.shape[2:]))))

    def forward(self, x):
        out = self.conv1(x + self.bias1a)
        out = F.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = out * self.scale + self.bias2b

        out = self.shortcut(x + self.bias1a) + out
        out = F.relu(out)

        return out


class FixupBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, n=4):
        super(FixupBottleneck, self).__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = PHConv(n, in_planes, planes, kernel_size=1, stride=1, bias=False)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = PHConv(n, planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.bias3a = nn.Parameter(torch.zeros(1))
        self.conv3 = PHConv(n, planes, self.expansion * planes, kernel_size=1, stride=1, bias=False)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias3b = nn.Parameter(torch.zeros(1))
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                PHConv(n, in_planes, self.expansion*planes, kernel_size=1, stride=stride)
            )
            nn.init.normal_(self.shortcut[0].F, mean=0, std=np.sqrt(2 / (self.shortcut[0].F.shape[0] * np.prod(self.shortcut[0].F.shape[2:]))))
            nn.init.normal_(self.shortcut[0].A, mean=0, std=np.sqrt(2 / (self.shortcut[0].A.shape[0] * np.prod(self.shortcut[0].A.shape[2:]))))

    def forward(self, x):
        identity = x

        out = self.conv1(x + self.bias1a)
        out = F.relu(out + self.bias1b)

        out = self.conv2(out + self.bias2a)
        out = F.relu(out + self.bias2b)

        out = self.conv3(out + self.bias3a)
        out = out * self.scale + self.bias3b

        out = self.shortcut(x + self.bias1a) + out
        out = F.relu(out)

        return out

class FixupResNet(nn.Module):
    def __init__(self, block, num_blocks, channels=4, n=4, num_classes=10):
        super(FixupResNet, self).__init__()

        self.num_layers = sum(num_blocks)
        self.in_planes = 64

        #self.conv1 = PHConv(n, channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = PHConv(n, channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, n=n)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, n=n)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, n=n)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, n=n)
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        nn.init.normal_(self.conv1.F, mean=0, std=np.sqrt(2 / (self.conv1.F.shape[0] * np.prod(self.conv1.F.shape[2:]))) * self.num_layers ** (-0.25))

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.F, mean=0, std=np.sqrt(2 / (m.conv1.F.shape[0] * np.prod(m.conv1.F.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.constant_(m.conv2.F, 0)
            elif isinstance(m, FixupBottleneck):
                nn.init.normal_(m.conv1.F, mean=0, std=np.sqrt(2 / (m.conv1.F.shape[0] * np.prod(m.conv1.F.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.normal_(m.conv2.F, mean=0, std=np.sqrt(2 / (m.conv2.F.shape[0] * np.prod(m.conv2.F.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.constant_(m.conv3.F, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.conv1(x) + self.bias1)
        #out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        #out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out + self.bias2)
        return out

class FixupResNetLarge(nn.Module):
    def __init__(self, block, num_blocks, channels=4, n=4, num_classes=10):
        super(FixupResNet, self).__init__()

        self.num_layers = sum(num_blocks)
        self.in_planes = 60

        self.conv1 = PHConv(n, channels, 60, kernel_size=3, stride=1, padding=1, bias=False)
        self.bias1 = nn.Parameter(torch.zeros(1))
        self.layer1 = self._make_layer(block, 60, num_blocks[0], stride=1, n=n)
        self.layer2 = self._make_layer(block, 120, num_blocks[1], stride=2, n=n)
        self.layer3 = self._make_layer(block, 240, num_blocks[2], stride=2, n=n)
        self.layer4 = self._make_layer(block, 516, num_blocks[3], stride=2, n=n)
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(516 * block.expansion, num_classes)

        nn.init.normal_(self.conv1.F, mean=0, std=np.sqrt(2 / (self.conv1.F.shape[0] * np.prod(self.conv1.F.shape[2:]))) * self.num_layers ** (-0.16))

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.F, mean=0, std=np.sqrt(2 / (m.conv1.F.shape[0] * np.prod(m.conv1.F.shape[2:]))) * self.num_layers ** (-0.25))
                nn.init.constant_(m.conv2.F, 0)
            elif isinstance(m, FixupBottleneck):
                nn.init.normal_(m.conv1.F, mean=0, std=np.sqrt(2 / (m.conv1.F.shape[0] * np.prod(m.conv1.F.shape[2:]))) * self.num_layers ** (-0.16))
                nn.init.normal_(m.conv2.F, mean=0, std=np.sqrt(2 / (m.conv2.F.shape[0] * np.prod(m.conv2.F.shape[2:]))) * self.num_layers ** (-0.16))
                nn.init.constant_(m.conv3.F, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        
    def _make_layer(self, block, planes, num_blocks, stride, n):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, n))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.conv1(x) + self.bias1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out + self.bias2)
        return out

def PHCResNet18(channels=4, n=4, num_classes=10):
    return FixupResNet(FixupBasicBlock, [2, 2, 2, 2], channels=channels, n=n, num_classes=num_classes)

def PHCResNet18Large(channels=4, n=4, num_classes=10):
    return FixupResNetLarge(FixupBasicBlock, [2, 2, 2, 2], channels=channels, n=n, num_classes=num_classes)

def PHCResNet50(channels=4, n=4, num_classes=10):
    return FixupResNet(FixupBottleneck, [3, 4, 6, 3], channels=channels, n=n, num_classes=num_classes)

def PHCResNet50Large(channels=4, n=4, num_classes=10):
    return FixupResNetLarge(FixupBottleneck, [3, 4, 6, 3], channels=channels, n=n, num_classes=num_classes)

def PHCResNet152(channels=4, n=4, num_classes=10):
    return FixupResNet(FixupBottleneck, [3, 8, 36, 3], channels=channels, n=n, num_classes=num_classes)

def PHCResNet152Large(channels=4, n=4, num_classes=10):
    return FixupResNetLarge(FixupBottleneck, [3, 8, 36, 3], channels=channels, n=n, num_classes=num_classes)
