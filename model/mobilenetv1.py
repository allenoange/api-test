'''MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
copyright: https://github.com/kuangliu/pytorch-cifar/blob/master/models/mobilenet.py
'''

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P


class Block(nn.Cell):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1, group=in_planes, has_bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=in_planes, momentum=0.9)
        self.conv2 = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=1, pad_mode='pad', padding=0, has_bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_planes, momentum=0.9)

    def construct(self, x):
        out = P.ReLU()(self.bn1(self.conv1(x)))
        out = P.ReLU()(self.bn2(self.conv2(out)))
        return out


class MobileNetV1(nn.Cell):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, pad_mode='pad', padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32, momentum=0.9)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Dense(in_channels=1024, out_channels=num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.SequentialCell([*layers])

    def construct(self, x):
        out = P.ReLU()(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = P.AvgPool(2, 2, 'valid')(out)
        out = P.Reshape()(out, (P.Shape()(out)[0], -1,))
        out = self.linear(out)
        return out
