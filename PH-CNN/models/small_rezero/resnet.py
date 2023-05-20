import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']
__some__ = ['resnet20']

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1, option='B', rezero=True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
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


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channels=3, rezero=False):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1, rezero=rezero)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2, rezero=rezero)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2, rezero=rezero)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

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
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class ResNetLarge(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channels=3, rezero=False):
        super(ResNetLarge, self).__init__()
        self.in_planes = 24

        self.conv1 = nn.Conv2d(channels, 24, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.layer1 = self._make_layer(block, 24, num_blocks[0], stride=1, rezero=rezero)
        self.layer2 = self._make_layer(block, 72, num_blocks[1], stride=2, rezero=rezero)
        self.layer3 = self._make_layer(block, 216, num_blocks[2], stride=2, rezero=rezero)
        self.linear = nn.Linear(216, num_classes)

        self.apply(_weights_init)

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
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20(channels=3, num_classes=10, rezero=False):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, channels=channels, rezero=rezero)

def resnet20large(channels=3, num_classes=10, rezero=False):
    return ResNetLarge(BasicBlock, [3, 3, 3], num_classes=num_classes, channels=channels, rezero=rezero)


def resnet32(channels=3, num_classes=10, rezero=False):
    return ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes, channels=channels, rezero=rezero)


def resnet44(channels=3, num_classes=10, rezero=False):
    return ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes, channels=channels, rezero=rezero)


def resnet56(channels=3, num_classes=10, rezero=False):
    return ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes, channels=channels, rezero=rezero)


def resnet110(channels=3, num_classes=10, rezero=False):
    return ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes, channels=channels, rezero=rezero)

def resnet110large(channels=3, num_classes=10, rezero=False):
    return ResNetLarge(BasicBlock, [18, 18, 18], num_classes=num_classes, channels=channels, rezero=rezero)


def resnet1202(channels=3, num_classes=10, rezero=False):
    return ResNet(BasicBlock, [200, 200, 200], num_classes=num_classes, channels=channels, rezero=rezero)
