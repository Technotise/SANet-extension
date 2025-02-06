from functools import partial
import torch
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from utils_bottleneck import Bottleneck

class P3D(nn.Module):
    def __init__(self, block, layers, factor, input_channel=3,
                 shortcut_type='B', num_classes=400, dropout=0.5, ST_struc=('A', 'B', 'C')):
        self.inplanes = 64
        self.factor = factor
        super(P3D, self).__init__()
        self.ST_struc = ST_struc

        self.conv1_custom = nn.Conv3d(input_channel, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                      padding=(0, 3, 3), bias=False)

        self.depth_3d = sum(layers[:3])

        self.bn1 = nn.BatchNorm3d(64)
        self.cnt = 0
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))
        self.maxpool_2 = nn.MaxPool3d(kernel_size=(2, 1, 1), padding=0,
                                      stride=(2, 1, 1))

        self.layer1 = self._make_layer(block, int(64*self.factor), layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, int(128*self.factor), layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, int(256*self.factor), layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, int(512*self.factor), layers[3], shortcut_type, stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=(5, 5), stride=1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(int(2048*self.factor) * 3 * 3, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        stride_p = stride

        if self.cnt < self.depth_3d:
            if self.cnt == 0:
                stride_p = 1
            else:
                stride_p = (1, 2, 2)
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv3d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=stride_p, bias=False),
                        nn.BatchNorm3d(planes * block.expansion)
                    )

        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                if shortcut_type == 'A':
                    downsample = partial(downsample_basic_block,
                                         planes=planes * block.expansion,
                                         stride=stride)
                else:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.inplanes, planes * block.expansion,
                                  kernel_size=1, stride=2, bias=False),
                        nn.BatchNorm2d(planes * block.expansion)
                    )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, n_s=self.cnt, depth_3d=self.depth_3d,
                            ST_struc=self.ST_struc))
        self.cnt += 1

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, n_s=self.cnt, depth_3d=self.depth_3d, ST_struc=self.ST_struc))
            self.cnt += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1_custom(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.maxpool_2(self.layer1(x))
        x = self.maxpool_2(self.layer2(x))
        x = self.maxpool_2(self.layer3(x))

        sizes = x.size()

        x = x.view(-1, sizes[1], sizes[3], sizes[4])
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(-1, int(2048*self.factor) * 3 * 3)
        x = self.fc(self.dropout(x))

        return x

def P3D63(**kwargs):
    model = P3D(Bottleneck, **kwargs)
    return model

def P3D131(**kwargs):
    model = P3D(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1),
                             out.size(2), out.size(3),
                             out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class P3DNetModified(P3D):
    def __init__(self, factor):
        super(P3DNetModified, self).__init__(
            factor=factor,
            block=Bottleneck,
            layers=[3, 4, 23, 3],
            input_channel=4,
            num_classes=4096,
            dropout=0.5,
            ST_struc=('A', 'B', 'C')
        )

        self.conv1_custom = nn.Conv3d(4, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                                      padding=(0, 3, 3), bias=False)

    def forward(self, x):
        return super(P3DNetModified, self).forward(x)
