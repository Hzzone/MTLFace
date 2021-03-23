import torch.utils.data
from torch.nn import functional as F
import torch
import torch.nn as nn

from common.ops import get_norm_layer, group2feature


class TaskRouter(nn.Module):

    def __init__(self, unit_count, age_group, sigma):
        super(TaskRouter, self).__init__()

        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)

        self.register_buffer('_unit_mapping', torch.zeros((age_group, conv_dim)))
        # self._unit_mapping = torch.zeros((age_group, conv_dim))
        start = 0
        for i in range(age_group):
            self._unit_mapping[i, start: start + unit_count] = 1
            start = int(start + (1 - sigma) * unit_count)

    def forward(self, inputs, task_ids):
        mask = torch.index_select(self._unit_mapping, 0, task_ids.long()) \
            .unsqueeze(2).unsqueeze(3)
        inputs = inputs * mask
        return inputs


class ResidualBlock(nn.Module):
    def __init__(self, unit_count, age_group, sigma):
        super(ResidualBlock, self).__init__()
        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)
        self.conv1 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, 1, 1), nn.BatchNorm2d(conv_dim))
        self.router1 = TaskRouter(unit_count, age_group, sigma)
        self.conv2 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, 1, 1), nn.BatchNorm2d(conv_dim))
        self.router2 = TaskRouter(unit_count, age_group, sigma)
        self.relu1 = nn.PReLU(conv_dim)
        self.relu2 = nn.PReLU(conv_dim)

    def forward(self, inputs):
        x, task_ids = inputs[0], inputs[1]
        residual = x
        x = self.router1(self.conv1(x), task_ids)
        x = self.relu1(x)
        x = self.router2(self.conv2(x), task_ids)
        return {0: self.relu2(residual + x), 1: task_ids}


class Upsample(nn.Module):
    def __init__(self, x_channels, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.PReLU(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.shortcut = nn.Conv2d(x_channels, out_channels, kernel_size=1)

    def forward(self, x, up):
        if x.size(2) < up.size(2):
            x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=False)

        p = self.conv(torch.cat([x, up], dim=1))
        sc = self.shortcut(x)

        p = p + sc

        p2 = self.conv2(p)

        return p + p2


class AgingModule(nn.Module):
    def __init__(self, age_group, repeat_num=4):
        super(AgingModule, self).__init__()
        layers = []
        sigma = 0.1
        unit_count = 128
        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, conv_dim, 1, 1, 0),
            nn.BatchNorm2d(conv_dim),
            nn.PReLU(conv_dim),
        )
        self.router = TaskRouter(unit_count, age_group, sigma)
        for _ in range(repeat_num):
            layers.append(ResidualBlock(unit_count, age_group, sigma))
        self.transform = nn.Sequential(*layers)
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_dim, 512, 1, 1, 0),
            nn.BatchNorm2d(512),
            nn.PReLU(512),
        )

        self.up_1 = Upsample(512, 512 + 256, 256)
        self.up_2 = Upsample(256, 256 + 128, 128)
        self.up_3 = Upsample(128, 128 + 64, 64)
        self.up_4 = Upsample(64, 64 + 3, 32)
        self.conv3 = nn.Conv2d(32, 3, 1, 1, 0)
        self.__init_weights()

    def forward(self, input_img, x_1, x_2, x_3, x_4, x_5, x_id, x_age, condition):
        x_id = self.conv1(x_id)
        x_id = self.router(x_id, condition)
        inputs = {0: x_id, 1: condition}
        x = self.transform(inputs)[0]
        x = self.conv2(x)
        x = self.up_1(x, x_4)
        x = self.up_2(x, x_3)
        x = self.up_3(x, x_2)
        x = self.up_4(x, input_img)
        x = self.conv3(x)
        return input_img + x

    def __init_weights(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                m.weight.data.normal_(0, 0.01)
                if hasattr(m.bias, 'data'):
                    m.bias.data.fill_(0)


class PatchDiscriminator(nn.Module):

    def __init__(self, age_group, conv_dim=64, repeat_num=3, norm_layer='bn'):
        super(PatchDiscriminator, self).__init__()

        use_bias = True
        self.age_group = age_group

        self.conv1 = nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1)
        sequence = []
        nf_mult = 1

        for n in range(1, repeat_num):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                get_norm_layer(norm_layer, nn.Conv2d(conv_dim * nf_mult_prev + (self.age_group if n == 1 else 0),
                                                     conv_dim * nf_mult, kernel_size=4, stride=2, padding=1,
                                                     bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** repeat_num, 8)

        sequence += [
            get_norm_layer(norm_layer,
                           nn.Conv2d(conv_dim * nf_mult_prev, conv_dim * nf_mult, kernel_size=4,
                                     stride=1, padding=1,
                                     bias=use_bias)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(conv_dim * nf_mult, 1, kernel_size=4, stride=1,
                      padding=1)  # output 1 channel prediction map
        ]
        self.main = nn.Sequential(*sequence)

    def forward(self, inputs, condition):
        x = F.leaky_relu(self.conv1(inputs), 0.2, inplace=True)
        condition = group2feature(condition, feature_size=x.size(2), age_group=self.age_group).to(x)
        return self.main(torch.cat([x, condition], dim=1))
