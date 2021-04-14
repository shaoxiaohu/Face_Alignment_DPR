import torch
import torch.nn as nn
import math


class GlobalReinitNet(nn.Module):
    def __init__(self):
        super(GlobalReinitNet, self).__init__()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2, padding=0),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0),
            nn.MaxPool2d(2, stride=2),
            nn.PReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.PReLU(),
            #nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            #nn.PReLU()
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(256, 32),
            nn.PReLU(),
            nn.Linear(32, 3 * 2)
        )

        self._initialize_weights()
        
        # Initialize the weights/bias with identity transformation
        #self.fc_loc[2].weight.data.zero_()
        #self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # transform the input

        xs = self.localization(x)
        xs = torch.flatten(xs, 1)
        out = self.fc_loc(xs)
        return  out 

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

class LocalReinitNet(nn.Module):
    def __init__(self, input_dim=196):
        super(LocalReinitNet, self).__init__()
        # Spatial transformer localization-network
        self.left_eye_net = self.make_net(input_dim)
        self.right_eye_net = self.make_net(input_dim)
        self.nose_net = self.make_net(input_dim)
        self.mouth_net = self.make_net(input_dim)

        self._initialize_weights()

        # Initialize the weights/bias with identity transformation

    def make_net(self, input_dim):
        backbone_net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.PReLU(),
            nn.Linear(128, 64),
            nn.PReLU(),
            nn.Linear(64, 3 * 2),
        )
        return  backbone_net
     
    def forward(self, x):
        #pdb.set_trace()
        out_1 = self.left_eye_net(x)

        out_2 = self.right_eye_net(x)

        out_3 = self.nose_net(x)

        out_4 = self.mouth_net(x)

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

