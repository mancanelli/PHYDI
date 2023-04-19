import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, rezero=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        self.rezero = rezero
        if self.rezero:
            self.res_weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.rezero:
            out = self.shortcut(x) + self.res_weight * F.relu(out)
        else:
            out = F.relu(self.shortcut(x) + out)
    
        return out


class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, rezero=True):
        super(Bottleneck, self).__init__()
    
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * 
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, 
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
    
        self.rezero = rezero
        if self.rezero:
            self.res_weight = nn.Parameter(torch.zeros(1), requires_grad = True)
  
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.rezero:
            out = self.shortcut(x) + self.res_weight * F.relu(out)
        else:
            out = F.relu(self.shortcut(x) + out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, rezero=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, rezero=rezero)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, rezero=rezero)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, rezero=rezero)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, rezero=rezero)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        #self.apply(He_init)

    def _make_layer(self, block, planes, num_blocks, stride, rezero=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rezero=rezero))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetLarge(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, rezero=False):
        super(ResNetLarge, self).__init__()
        self.in_planes = 60

        self.conv1 = nn.Conv2d(3, 60, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(60)
        self.layer1 = self._make_layer(block, 60, num_blocks[0], stride=1, rezero=rezero)
        self.layer2 = self._make_layer(block, 120, num_blocks[1], stride=2, rezero=rezero)
        self.layer3 = self._make_layer(block, 240, num_blocks[2], stride=2, rezero=rezero)
        self.layer4 = self._make_layer(block, 516, num_blocks[3], stride=2, rezero=rezero)
        self.linear = nn.Linear(516*block.expansion, num_classes)

        #self.apply(He_init)

    def _make_layer(self, block, planes, num_blocks, stride, rezero=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, rezero=rezero))
            self.in_planes = planes * block.expansion
        
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10, rezero=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, rezero=rezero)

def ResNet18Large(num_classes=10, rezero=False):
    return ResNetLarge(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, rezero=rezero)


def ResNet34(num_classes=10, rezero=False):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, rezero=rezero)


def ResNet50(num_classes=10, rezero=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, rezero=rezero)

def ResNet50Large(num_classes=10, rezero=False):
    return ResNetLarge(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, rezero=rezero)


def ResNet101(num_classes=10, rezero=False):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, rezero=rezero)


def ResNet152(num_classes=10, rezero=False):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, rezero=rezero)

def ResNet152Large(num_classes=10, rezero=False):
    return ResNetLarge(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, rezero=rezero)