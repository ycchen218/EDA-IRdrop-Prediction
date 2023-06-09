import torch.nn as nn
import torch.nn.functional as F
from CoordConv import CoordConv2d


class SCE(nn.Module):
    #https://github.com/RooKichenn/CEFPN/blob/main/backbone/feature_pyramid_network.py
    def __init__(self, in_channels):
        super(SCE, self).__init__()

        self.conv3x3 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1x1_2 = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)
        self.pixel_shuffle_4 = nn.PixelShuffle(upscale_factor=4)

        self.globalpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1x1_3 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)

    def forward(self, x):
        out_size = x.shape[-2:]
        out_size = [x*2 for x in out_size]
        branch1 = self.pixel_shuffle(self.conv3x3(x))
        branch2 = F.interpolate(self.pixel_shuffle_4(self.conv1x1_2(self.maxpool(x))), size=out_size, mode="nearest")
        branch3 = self.conv1x1_3(self.globalpool(x))
        out = (branch1 + branch2 + branch3)
        return out


class CAG(nn.Module):
    #https://github.com/RooKichenn/CEFPN/blob/main/backbone/feature_pyramid_network.py
    def __init__(self, in_channels):
        super(CAG, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.fc1 = nn.Conv2d(in_channels, in_channels, 1)
        self.fc2 = nn.Conv2d(in_channels, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        fc1 = self.relu(self.fc1(self.avgpool(x)))
        fc2 = self.relu(self.fc2(self.maxpool(x)))
        out = fc1 + fc2
        return self.sigmoid(out)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CEFPN(nn.Module):
    def __init__(self, block, num_blocks,in_channel, device):
        super(CEFPN, self).__init__()

        self.in_planes = 64
        if device=='cpu':
            self.conv1 = CoordConv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False,use_cuda=False)
        else:
            self.conv1 = CoordConv2d(in_channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels

        #CE
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.SSF_c5 = nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0)
        self.SSF_c4 = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)
        self.conv_c4 = nn.Conv2d(1024,256,kernel_size=1,stride=1,padding=0)
        self.conv_c3 = nn.Conv2d(512,256,kernel_size=1,stride=1,padding=0)
        self.conv_c2 = nn.Conv2d(256,256,kernel_size=1,stride=1,padding=0)
        self.SCE = SCE(in_channels=2048)
        self.CAG = CAG(in_channels=256)

        # Smooth layers
        #todo smooth 256-->128
        self.smooth1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

        #Gobal
        self.Adpool = nn.AdaptiveMaxPool2d(output_size=(1,1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()

        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1) #(bs,64,128,128)
        c2 = self.layer1(c1) #(bs,256,128,128)
        c3 = self.layer2(c2)#(bs,512,64,64)
        c4 = self.layer3(c3)#(bs,1024,32,32)
        c5 = self.layer4(c4)#(bs,2048,16,16)
        #CE
        SCE_out = self.SCE(c5) #(bs,256,32,32)
        f4 = self.SSF_c5(self.pixel_shuffle(c5))+self.conv_c4(c4) #(bs, 256, 32, 32)
        f3 = self.SSF_c4(self.pixel_shuffle(c4))+self.conv_c3(c3) #(bs, 256, 64, 64)
        f2 = self.conv_c2(c2) #(bs, 256, 128, 128)
        # Top-down
        p4 = f4 #(bs, 256, 32, 32)
        p3 = self._upsample_add(p4,f3) #(bs, 256, 64, 64)
        p2 = self._upsample_add(p3,f2) #(bs, 256, 128, 128)

        # make r
        out_size = p4.shape[-2:]
        SCE_out = F.interpolate(SCE_out, size=out_size, mode='bilinear')
        I_p4 = F.interpolate(p4, size=out_size, mode='bilinear')
        I_p3 = F.adaptive_max_pool2d(p3, output_size=out_size)
        I_p2 = F.adaptive_max_pool2d(p2, output_size=out_size)

        I = (I_p4 + I_p3 + I_p2 + SCE_out) / 4
        CA = self.CAG(I)

        r5 = F.adaptive_max_pool2d(I, output_size=c5.shape[-2:])
        r5 = r5 * CA
        residual_r4 = F.adaptive_max_pool2d(I, output_size=c4.shape[-2:])
        r4 = (residual_r4 + f4) * CA
        residual_r3 = F.interpolate(I, size=c3.shape[-2:], mode='bilinear')
        r3 = (residual_r3 + f3) * CA
        residual_r2 = F.interpolate(I, size=c2.shape[-2:], mode='bilinear')
        r2 = (residual_r2 + f2) * CA

        # smooth
        r2 = self.smooth1(r2)
        r3 = self.smooth2(r3)
        r4 = self.smooth3(r4)
        r5 = self.Adpool(r5)
        return r2, r3, r4, r5
def CEFPN101(in_channel,device):
    return CEFPN(Bottleneck, [2,2,2,2],in_channel, device)
