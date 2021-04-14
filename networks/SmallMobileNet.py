
from __future__ import division
import torch
import torch.nn as nn
import math

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SmallMobileNetV2(nn.Module):
    def __init__(self, widen_factor=1.0, num_classes=1000): #, input_size=224
        super(SmallMobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 8
        last_channel = 64
        interverted_residual_setting = [
            # t, c, n, s
            [1, 8, 1, 1],
            [6, 12, 2, 2],
            [6, 16, 2, 2],
            [6, 24, 3, 2],
            [6, 32, 3, 2],
            [6, 48, 3, 2],
            [6, 64, 2, 2],
            [6, 80, 1, 1],
        ]

        # building first layer
        # assert input_size % 32 == 0
        input_channel = int(input_channel * widen_factor)
        self.last_channel = int(last_channel * widen_factor) if widen_factor > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * widen_factor)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # building pts net
        self.pts_net = nn.Sequential(
            nn.Linear(4*self.last_channel, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, num_classes)
        )


        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        pts = self.pts_net(x)
        return pts

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()





class SmallMobileNetV2Part(nn.Module):
    def __init__(self, widen_factor=1.0, num_classes=68*2): #, input_size=224
        super(SmallMobileNetV2Part, self).__init__()
        self.block = InvertedResidual
        self.input_channel = 8
        self.last_channel = 64
        self.interverted_residual_setting = [
            # t, c, n, s
            [1, 8, 1, 1],
            [6, 12, 2, 2],
            [6, 16, 2, 2],
            [6, 24, 3, 2],
            [6, 32, 3, 2],
            [6, 48, 3, 2],
            [6, 64, 2, 2],
            [6, 80, 1, 1],
        ]

        if num_classes==68*2:
            part_dim = [22, 22, 18, 40]
        elif num_classes==98*2:
            part_dim = [36, 36, 18, 40]
        elif num_classes==106*2:
            part_dim = [38, 38, 30, 40]

        # building first layer
        # assert input_size % 32 == 0
        # Spatial transformer localization-network
        self.left_eye_net = self.make_backbone(widen_factor)
        self.right_eye_net = self.make_backbone(widen_factor)
        self.nose_net = self.make_backbone(widen_factor)
        self.mouth_net = self.make_backbone(widen_factor)
        # Regressor for the 3 * 2 affine matrix
        self.left_eye_loc = self.make_pts_fc(part_dim[0])
        self.right_eye_loc = self.make_pts_fc(part_dim[1])
        self.nose_loc = self.make_pts_fc(part_dim[2])
        self.mouth_loc = self.make_pts_fc(part_dim[3])

        self._initialize_weights()

    def make_backbone(self, widen_factor):
        # building first layer
        # assert input_size % 32 == 0
        input_channel = int(self.input_channel * widen_factor)
        last_channel = int(self.last_channel * widen_factor) if widen_factor > 1.0 else self.last_channel
        features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * widen_factor)
            for i in range(n):
                if i == 0:
                    features.append(self.block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    features.append(self.block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(conv_1x1_bn(input_channel, last_channel))
        # make it nn.Sequential
        return nn.Sequential(*features)

    def make_pts_fc(self,num_classes):
        #pdb.set_trace()
        pts_net = nn.Sequential(
            nn.Linear(self.last_channel, 64),
            nn.PReLU(),
            nn.Linear(64, 64),
            nn.PReLU(),
            nn.Linear(64, num_classes)
        )
        return pts_net


    def forward(self, x):
        xs_1 = self.left_eye_net(x[0])
        xs_1 = torch.flatten(xs_1, 1)
        #pdb.set_trace()
        out_1 = self.left_eye_loc(xs_1)

        xs_2 = self.right_eye_net(x[1])
        xs_2 = torch.flatten(xs_2, 1)
        out_2 = self.right_eye_loc(xs_2)

        xs_3 = self.nose_net(x[2])
        xs_3 = torch.flatten(xs_3, 1)
        out_3 = self.nose_loc(xs_3)

        xs_4 = self.mouth_net(x[3])
        xs_4 = torch.flatten(xs_4, 1)
        out_4 = self.mouth_loc(xs_4)

        return  [out_1, out_2, out_3, out_4]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

