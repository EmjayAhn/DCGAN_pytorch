# Generator

import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim=256, img_size=128, conv_dim=64):
        super(Generator, self).__init__()

        self.convt1 = nn.ConvTranspose2d(z_dim, conv_dim*8, int(img_size/16), stride=1, pad=0)
        self.bn1 = nn.BatchNorm2d(conv_dim*8)
        
        self.convt2 = nn.ConvTranspose2d(conv_dim*8, conv_dim*4, 4)
        self.bn2 = nn.BatchNorm2d(conv_dim * 4)

        self.convt3 = nn.ConvTranspose2d(conv_dim*4, conv_dim*2, 4)
        self.bn3 = nn.BatchNorm2d(conv_dim * 2)

        self.convt4 = nn.ConvTranspose2d(conv_dim*2, conv_dim, 4)
        self.bn4 = nn.BatchNorm2d(conv_dim)

        self.convt5 = nn.ConvTranspose2d(conv_dim, 3, 4)

        
    def forward(self, z):
        input = z.view(z.size(0), z.size(1), 1, 1)
        out = F.relu(self.bn1(self.convt1(input)))
        out = F.relu(self.bn2(self.convt2(out)))
        out = F.relu(self.bn3(self.convt3(out)))
        out = F.relu(self.bn4(self.convt4(out)))
        out = F.tanh(self.convt5(out))
        return out
