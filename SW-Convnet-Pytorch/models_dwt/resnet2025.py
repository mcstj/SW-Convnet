# This code is built from the PyTorch examples repository: https://github.com/pytorch/vision/tree/master/torchvision/models.
# Copyright (c) 2017 Torch Contributors.
# The Pytorch examples are available under the BSD 3-Clause License.
#
# ==========================================================================================
#
# Adobe’s modifications are Copyright 2019 Adobe. All rights reserved.
# Adobe’s modifications are licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License (CC-NC-SA-4.0). To view a copy of the license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
#
# ==========================================================================================
#
# BSD-3 License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

from models_dwt import *
#from kymatio import Scattering2D
#from kymatio.torch import Scattering2D
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D
import numpy as np
import torch
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d']


model_urls = {
     'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
     'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
     'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
     'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
     'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
 }


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                 padding=1, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# 定义小波函数 (Morlet小波)
def morlet_wavelet(t):
    t_cpu=t.cpu()
    #y=torch.exp(1j * 2.5 * t_cpu) * torch.exp(-0.5 * t_cpu**2)
    #y=2/np.sqrt(3)*np.pi**(-0.25)*(1-t_cpu**2)* torch.exp(-0.5 * t_cpu**2)
    #print(t_cpu.shape)
    y=torch.zeros(len(t_cpu))
    a=t_cpu>=0
    b=t_cpu<0.5
    y=(a*b).type(torch.float32)
    c=t_cpu>=0.5
    d=t_cpu<1
    y2=(-1)*(c*d).type(torch.float32)
    y=y+y2
    
    # for i in range(len(t_cpu)):
    #   if t_cpu[i]>=0 and t_cpu[i]<0.5 :
    #     y[i]=1
    #   elif t_cpu[i]>=0.5 and t_cpu[i]<1:
    #     y[i]=-1
    #   else:
    #     y[i]=0

    #y=y.float()
    return y.cuda()

# 定义小波激活函数层
class WaveletActivation(nn.Module):
    def __init__(self, in_features,sx,sy):
        super(WaveletActivation, self).__init__()
        # 可训练的伸缩因子和平移因子
        self.a0 = nn.Parameter(torch.randn(in_features,sx,sy))
        self.b0 = nn.Parameter(torch.randn(in_features,sx,sy))
        print("a shape=",self.a0.shape)
        print("b shape=",self.b0.shape)
        
    def forward(self, x):
        # 应用小波变换: (x-b)/a
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #x=x.cpu()
        #print("x shape=",x.shape,x.device)
        
        #print("a shape=",self.a.shape)
        #print("b shape=",self.b.shape)
        #self.b = torch.randn(x.shape[-3:])
        #self.a = torch.randn(x.shape[-3:])
        #self.b=self.b.expand(self.a.shape[0],x.shape[-2],x.shape[-1])
        
        #print("b shape=",self.b.shape,self.b.device)
        t = (x - self.b0) / self.a0
        # 使用Morlet小波的实部作为激活值
        #t=torch.real(morlet_wavelet(t))
        #print('t type:',type(t))
        t=morlet_wavelet(t)
        #print(t)
        #return t.float().to(device)
        return t

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, wavename = 'haar'):
        super(BasicBlock, self).__init__()
        #print('inplanes:',inplanes,'planes:',planes)
        J=1
        L_ang=4
        max_order=2
        scattering1 = ScatteringTorch2D(J=J, shape=(224, 224), L=L_ang, max_order=max_order,backend='torch')
        scattering1 = nn.Sequential(scattering1, nn.Flatten(1, 2))
        if max_order==2:
          K1=1+L_ang*J+int(L_ang*L_ang*J*(J-1)/2)
        else:
          K1=1+L_ang*J
        print('basic block k=',K1)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        print('planes=',planes)
        #self.inplanes = planes[0]
        self.s1=scattering1

        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        #self.relu = nn.ReLU(inplace=True)
        #self.relu = nn.PReLU(num_parameters=1, init=0.25)
        #self.relu = torch.nn.SiLU()
        self.relu = torch.nn.Mish()
        print("planes=",planes)
        #self.wavelet = WaveletActivation(planes)
        if(stride==1):
            self.conv2 = conv3x3(planes,planes)
        else:
            self.conv2 = nn.Sequential(Downsample(wavename = wavename),
                conv3x3(planes, planes),)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if 'xs1' in dir():
            pass   
        else:
            xs1=self.s1(x)

        print('Basic block x conv shape:',x.shape)
        out = self.conv1(x)
        print('Basic block x conv shape:',x.shape)
        out = self.bn1(out)
        out = self.relu(out)
        #out=self.wavelet(out)
        #print("out shape=",out.shape)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        #out=self.wavelet(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, wavename = 'haar'):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, groups) # stride moved
        self.bn2 = norm_layer(planes)
        if(stride==1):
            self.conv3 = conv1x1(planes, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(Downsample(wavename = wavename),
                conv1x1(planes, planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, wavename = 'haar', pool_only = True):
        super(ResNet, self).__init__()
        J=1
        L_ang=4
        max_order=2
        scattering1 = ScatteringTorch2D(J=J, shape=(224, 224), L=L_ang, max_order=max_order,backend='torch')
        scattering1 = nn.Sequential(scattering1, nn.Flatten(1, 2))
        if max_order==2:
          K1=1+L_ang*J+int(L_ang*L_ang*J*(J-1)/2)
        else:
          K1=1+L_ang*J
        print('k=',K1)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        print('planes=',planes)
        self.inplanes = planes[0]
        self.s1=scattering1

        """ J=2
        L_ang=2
        max_order=2
        scattering2 = ScatteringTorch2D(J=J, shape=(224, 224), L=L_ang, max_order=max_order,backend='torch')
        scattering2 = nn.Sequential(scattering2, nn.Flatten(1, 2))
        if max_order==2:
          K2=1+L_ang*J+int(L_ang*L_ang*J*(J-1)/2)
        else:
          K2=1+L_ang*J
        print('k2=',K2)
        self.inplanes = planes[0]
        self.s2=scattering2 """

        if(pool_only):
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=1, padding=3, bias=False) 
        '''if(pool_only):
            self.conv1 = nn.Conv2d(3*K, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3*K, planes[0], kernel_size=7, stride=1, padding=3, bias=False)
        '''
        self.bn1 = norm_layer(planes[0]+K1*3)
        # self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)
        print("planes[0]+K1*3=",planes[0]+K1*3)
        #self.wavelet = WaveletActivation(planes[0]+K1*3)
        self.wavelet = WaveletActivation(planes[0]+K1*3,112,112)

        if(pool_only):
            self.maxpool = nn.Sequential(*[Downsample(wavename = wavename)])
        else:
            self.maxpool = nn.Sequential(*[Downsample(wavename = wavename), Downsample(wavename = wavename)])
        
        # planes[0]+=(K1+K2)*3
        planes[0]+=K1*3
        self.inplanes = planes[0]
        #planes[0]+=K*3
        
        self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, wavename = wavename)
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer, wavename = wavename)
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer, wavename = wavename)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(planes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, wavename = 'haar'):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride, filter_size=filter_size),
            #     norm_layer(planes * block.expansion),
            # )

            downsample = [Downsample(wavename = wavename),] if(stride == 2) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
                norm_layer(planes * block.expansion)]
            # print(downsample)
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer, wavename = wavename))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer, wavename = wavename))

        return nn.Sequential(*layers)

    def forward(self, x):
        x0=x
        # #print("x shape=",x.shape)
        if 'xs1' in dir():
            pass   
        else:
            xs1=self.s1(x)
        # print("x1 shape=",x.shape)
        #print("xs shape=",xs1.shape)
        x = self.conv1(x)
        #print("x1  shape=",x.shape)
        x = torch.cat((x, xs1), dim=1)
        #print("x2 shape=",x.shape)
        x = self.bn1(x)
        #print("x3 shape=",x.shape)
        x = self.relu(x)
        #x = self.wavelet(x)
        # print("x4 shape=",x.shape)
        x = self.maxpool(x)
        # print("x5 shape=",x.shape)
        #xs2=self.s2(x0)
        # print("xs2 shape=",xs2.shape)
        #x = torch.cat((x, xs2), dim=1)
        # print("x5a shape=",x.shape)
        x = self.layer1(x)
        # print("x6 shape=",x.shape)
        x = self.layer2(x)
        # print("x6a shape=",x.shape)
        x = self.layer3(x)
        # print("x6b shape=",x.shape)
        x = self.layer4(x)
        # print("x7 shape=",x.shape)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print("x8 shape=",x.shape)
        x = self.fc(x)
        #print("x9 shape=",x.shape)

        return x


def resnet18(wavename = 'haar', pool_only=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], wavename = wavename, pool_only=pool_only, **kwargs)
    return model


def resnet34(wavename = 'haar', pool_only=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], wavename = wavename, pool_only=pool_only, **kwargs)
    return model


def resnet50(wavename = 'haar', pool_only=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], wavename = wavename, pool_only=pool_only, **kwargs)
    return model


def resnet101(wavename = 'haar', pool_only=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], wavename = wavename, pool_only=pool_only, **kwargs)
    return model


def resnet152(wavename = 'haar', pool_only=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], wavename = wavename, pool_only=pool_only, **kwargs)
    return model


def resnext50_32x4d(wavename = 'haar', pool_only=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], groups=4, width_per_group=32, wavename = wavename, pool_only=pool_only, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnext101_32x8d(wavename = 'haar', pool_only=True, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], groups=8, width_per_group=32, wavename = wavename, pool_only=pool_only, **kwargs)
    # if pretrained:
    #     model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


