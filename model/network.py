
"""
paper: Pyramid Channel-based Feature Attention Network for image dehazing 
file: network.py
about: model for PCFAN
author: Tao Wang
date: 06/13/20
"""


# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Channel Attention (CA) Layer --- #
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y 


# --- Main model  --- #
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #Conv1
        self.layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1)
            )
        #Conv2
        self.layer5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.layer6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        self.layer7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1)
            )
        #Conv3
        self.layer9 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.layer10 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.layer11 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
            )
        self.Attention_1 = CALayer(channel=32)
        self.Attention_2 = CALayer(channel=64)
        self.Attention_3 = CALayer(channel=128)
        self.Attention_4 = CALayer(channel=192)
        self.final = nn.Conv2d(224,3,kernel_size=3,padding=1)
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        #Conv1
        B,C,H,W = x.size()
        x0 = x
        x = self.layer1(x)
        x = self.layer2(x) + x
        map1 = self.layer3(x) + x     

        #Conv2
        map2 = self.layer5(map1)
        map2 = self.layer6(map2) + map2
        map2 = self.layer7(map2) + map2 
        map2 = nn.UpsamplingBilinear2d((H//2,W//2))(map2)
        #Conv3
        map3 = self.layer9(map2)    
        map3 = self.layer10(map3) + map3
        map3 = self.layer11(map3) + map3 

        map1 = self.Attention_1(map1)

        map2 = self.Attention_2(map2)
        map3 = self.Attention_3(map3)
        map3 = nn.UpsamplingBilinear2d((H//2,W//2))(map3)
        map23 = torch.cat((map2,map3),1)
        map23 = self.Attention_4(map23)
        map23 = nn.UpsamplingBilinear2d((H,W))(map23)

        map123 = torch.cat((map23,map1),1)
        out = self.tanh(self.final(map123))
        out = torch.add(x0,out)
        return out



