# Discriminator

import torch.nn as nn
import torch.nn.functional as F

def conv_custom(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """
    c_in: number of input channel
    c_out: number of output channel
    k_size: kernel size
    
    bn: True for append Batch Normalization layer.
    """
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    def __init__(self, image_size=128, conv_dim=64):
        super(Discriminator, self).__init__()

        self.conv1 = conv_custom(3, conv_dim, 4, bn=False)
        self.conv2 = conv_custom(conv_dim, conv_dim*2, 4)
        self.conv3 = conv_custom(conv_dim*2, conv_dim*4, 4)
        self.conv4 = conv_custom(conv_dim*4, conv_dim*8, 4)
        self.conv5 = conv_custom(conv_din*8, 1, int(image / 16), 1, 0, False)

    def forward(self, img):
        out = F.leaky_relu(self.conv1(img), negative_slope=0.2)
        out = F.leaky_relu(self.conv2(img), negative_slope=0.2)
        out = F.leaky_relu(self.conv3(img), negative_slope=0.2)
        out = F.leaky_relu(self.conv4(img), negative_slope=0.2)
        out = F.sigmoid(self.conv5(img), negative_slope=0.2)
        out = out.squeeze()
        return out