import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math

norm_layer = nn.InstanceNorm2d
act = lambda x: nn.LeakyReLU(0.2, True)


class TaskRouter(nn.Module):

    def __init__(self, unit_count, age_group, sigma):
        super(TaskRouter, self).__init__()

        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)

        self.register_buffer('_unit_mapping', torch.zeros((age_group, conv_dim)))
        start = 0
        for i in range(age_group):
            self._unit_mapping[i, start: start + unit_count] = 1
            start = int(start + (1 - sigma) * unit_count)

    def forward(self, inputs, task_ids):
        mask = torch.index_select(self._unit_mapping, 0, task_ids.long()) \
            .unsqueeze(2).unsqueeze(3)
        inputs = inputs * mask
        return inputs


class StyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super(StyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True)]
        for i in range(num_pools - 1):
            modules += [
                nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        modules.append(nn.AdaptiveAvgPool2d(1))
        self.convs = nn.Sequential(*modules)
        self.linear = nn.Linear(out_c, out_c)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, unit_count, age_group, sigma):
        super(ResidualBlock, self).__init__()
        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)
        self.conv1 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, 1, 1), norm_layer(conv_dim))
        self.router1 = TaskRouter(unit_count, age_group, sigma)
        self.conv2 = nn.Sequential(nn.Conv2d(conv_dim, conv_dim, 3, 1, 1), norm_layer(conv_dim))
        self.router2 = TaskRouter(unit_count, age_group, sigma)
        self.relu1 = act(conv_dim)
        self.relu2 = act(conv_dim)

    def forward(self, inputs):
        x, task_ids = inputs[0], inputs[1]
        residual = x
        x = self.router1(self.conv1(x), task_ids)
        x = self.relu1(x)
        x = self.router2(self.conv2(x), task_ids)
        return {0: self.relu2(residual + x), 1: task_ids}


class AgingModule(nn.Module):
    def __init__(self, in_channel, age_group, repeat_num=4):
        super(AgingModule, self).__init__()
        layers = []
        sigma = 1 / 8
        unit_count = in_channel // 4
        conv_dim = int((age_group - (age_group - 1) * sigma) * unit_count)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, conv_dim, 1, 1, 0),
            norm_layer(conv_dim),
            act(conv_dim),
        )
        self.router = TaskRouter(unit_count, age_group, sigma)
        for _ in range(repeat_num):
            layers.append(ResidualBlock(unit_count, age_group, sigma))
        self.transform = nn.Sequential(*layers)
        self.conv2 = nn.Sequential(
            nn.Conv2d(conv_dim, in_channel, 1, 1, 0),
        )

    def forward(self, x_id, condition):
        x_id = self.conv1(x_id)
        x_id = self.router(x_id, condition)
        inputs = {0: x_id, 1: condition}
        x = self.transform(inputs)[0]
        x = self.conv2(x)
        return x


class Upsample(nn.Module):
    def __init__(self, x_channels, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            norm_layer(in_channels),
            act(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            norm_layer(out_channels),
            act(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.conv2 = nn.Sequential(
            norm_layer(out_channels),
            act(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            norm_layer(out_channels),
            act(out_channels),
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


class SPPModule(nn.Module):
    def __init__(self, pool_mode='avg', sizes=(1, 2, 3, 6)):
        super().__init__()
        if pool_mode == 'avg':
            pool_layer = nn.AdaptiveAvgPool2d
        elif pool_mode == 'max':
            pool_layer = nn.AdaptiveMaxPool2d
        else:
            raise NotImplementedError

        self.pool_blocks = nn.ModuleList([
            nn.Sequential(pool_layer(size), nn.Flatten()) for size in sizes
        ])

    def forward(self, x):
        xs = [block(x) for block in self.pool_blocks]
        x = torch.cat(xs, dim=1)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return x


class AttentionModule(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super(AttentionModule, self).__init__()
        kernel_size = 7
        pool_size = (1, 2, 3)
        self.avg_spp = SPPModule('avg', pool_size)
        self.max_spp = SPPModule('max', pool_size)
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2,
                      dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )

        _channels = channels * int(sum([x ** 2 for x in pool_size])) * 2
        self.channel = nn.Sequential(
            nn.Conv2d(
                _channels, _channels // reduction, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                _channels // reduction, channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01, affine=True),
            nn.Sigmoid()
        )
        self.norm = nn.Identity()
        self.act = nn.Identity()

    def forward(self, x):
        channel_input = torch.cat([self.avg_spp(x), self.max_spp(x)], dim=1)
        channel_scale = self.channel(channel_input)

        spatial_input = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        spatial_scale = self.spatial(spatial_input)
        scale = (channel_scale + spatial_scale) * 0.5

        x_id = x * scale
        x_id = self.act(self.norm(x_id))
        x_age = x - x_id

        return x_id, x_age


class Encoder(nn.Module):
    def __init__(self,
                 age_group,
                 repeat_num=4,
                 input_size=112):
        super(Encoder, self).__init__()
        self.n_styles = math.ceil(math.log(input_size, 2)) * 2 - 2
        facenet = IR_50(input_size=input_size)
        self.input_layer = facenet.input_layer
        self.block1 = facenet.body[0]
        self.block2 = facenet.body[1]
        self.block3 = facenet.body[2]
        self.block4 = facenet.body[3]
        self.output_layer = nn.Sequential(facenet.bn2, nn.Flatten(), facenet.dropout, facenet.fc, facenet.features)
        self.fsm = AttentionModule()

        self.styles = nn.ModuleList()
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.n_styles):
            if i < self.coarse_ind:
                style = StyleBlock(512, 512, input_size // 16)
            elif i < self.middle_ind:
                style = StyleBlock(512, 512, input_size // 8)
            else:
                style = StyleBlock(512, 512, input_size // 2)
            self.styles.append(style)
        self.upsample1 = Upsample(512, 512 + 256, 256)
        self.upsample2 = Upsample(256, 256 + 128, 128)
        self.upsample3 = Upsample(128, 128 + 64, 64)
        self.latlayer1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.latlayer2 = nn.Conv2d(64, 512, 3, 1, 1)
        self.aging1 = AgingModule(512, age_group, repeat_num)
        self.aging2 = AgingModule(256, age_group, repeat_num)
        self.aging3 = AgingModule(128, age_group, repeat_num)

    def encode(self, x):
        x = self.input_layer(x)
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        x = self.block4(c3)
        x_id, x_age = self.fsm(x)
        x_vec = F.normalize(self.output_layer(x_id), dim=1)
        return x_id, x_vec, x_age, c1, c2, c3

    def get_conditions(self, c1, c2, c3, x_id, conditions):
        latents = []
        for j in range(self.coarse_ind):
            latents.append(self.styles[j](x_id))

        c4 = self.aging1(x_id, conditions)
        p3 = self.upsample1(c4, c3)

        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](self.latlayer1(p3)))

        p3 = self.aging2(p3, conditions)
        p4 = self.upsample2(p3, c2)
        p4 = self.aging3(p4, conditions)
        p4 = self.upsample3(p4, c1)
        for j in range(self.middle_ind, self.n_styles):
            latents.append(self.styles[j](self.latlayer2(p4)))

        latents = torch.stack(latents, dim=1)
        return latents

    def forward(self, x, conditions=None):
        x_id, x_vec, x_age, c1, c2, c3 = self.encode(x)

        if conditions is not None:
            latents = self.get_conditions(c1, c2, c3, x_id, conditions)
            return latents, x_vec, x_age
        return x_vec, x_age


class MTLFace(nn.Module):
    def __init__(self):
        super(MTLFace, self).__init__()
        input_size = 112
        age_group = 7
        self.decoder = Decoder(input_size=input_size)
        self.encoder = Encoder(age_group=7,
                               repeat_num=4,
                               input_size=input_size)
        self.age_estimator = AgeEstimationModule(input_size, age_group)

    def encode(self, x):
        _, x_vec1, x_age1, _, _, _ = self.encoder.encode(x)
        _, x_vec2, x_age2, _, _, _ = self.encoder.encode(torch.flip(x, dims=(3,)))
        x_vec = F.normalize(x_vec1 + x_vec2, dim=1)
        x_age1, x_group1 = self.age_estimator(x_age1)
        x_age2, x_group2 = self.age_estimator(x_age2)
        x_age, x_group = (x_age1 + x_age2) * 0.5, (x_group1 + x_group2) * 0.5
        return x_vec, x_age

    def get_conditions(self, x, conditions):
        x_id, x_vec, x_age, c1, c2, c3 = self.encoder.encode(x)
        latents = self.encoder.get_conditions(c1, c2, c3, x_id, conditions)
        return latents

    def aging(self, x, conditions):
        latents = self.get_conditions(x, conditions)
        outputs = self.decoder(latents).clamp(-1, 1.)
        return outputs


class Decoder(nn.Module):

    def __init__(self, input_size):
        super(Decoder, self).__init__()

        self.input_size = input_size
        base = math.ceil(math.log(input_size, 2))
        from .stylegan2.model import Generator as StyleGAN2Generator
        self.decoder = StyleGAN2Generator(2 ** base, style_dim=512, n_mlp=8)
        self.register_buffer('latent_avg', torch.zeros(1, 1, 512))

    def forward(self, codes):
        codes = codes + self.latent_avg

        images, _ = self.decoder([codes],
                                 input_is_latent=True,
                                 randomize_noise=True,
                                 return_latents=False)
        images = F.interpolate(images, size=self.input_size, mode='bilinear', align_corners=True)

        return images


class AgeEstimationModule(nn.Module):
    def __init__(self, input_size, age_group):
        super(AgeEstimationModule, self).__init__()
        self.out_neurons = 101
        self.age_group = age_group
        self.age_output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Flatten(),
            nn.Linear(512 * (input_size // 16) ** 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.out_neurons),
        )
        self.group_output_layer = nn.Linear(self.out_neurons, age_group)
        self.register_buffer('expectation', torch.arange(self.out_neurons).float())

    def forward(self, x_age):
        x_age = self.age_output_layer(x_age)
        x_group = self.group_output_layer(x_age)
        x_age = (F.softmax(x_age, dim=1) * self.expectation[None, :]).sum(dim=1)
        return x_age, x_group


class bottleneck_IR(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False), nn.BatchNorm2d(depth))
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


def get_block(unit_module, in_channel, depth, num_units, stride=2):
    layers = [unit_module(in_channel, depth, stride)] + [unit_module(depth, depth, 1) for _ in range(num_units - 1)]
    return nn.Sequential(*layers)


class IResNet(nn.Module):
    dropout_ratio = 0.4

    def __init__(self, input_size, num_layers, mode='ir', amp=False):
        super(IResNet, self).__init__()
        assert mode in ['ir', 'ir_se'], "mode should be ir or ir_se"

        if mode == 'ir':
            unit_module = bottleneck_IR
        else:
            raise NotImplementedError

        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                         nn.BatchNorm2d(64),
                                         nn.PReLU(64))

        block1 = get_block(unit_module, in_channel=64, depth=64, num_units=num_layers[0])
        block2 = get_block(unit_module, in_channel=64, depth=128, num_units=num_layers[1])
        block3 = get_block(unit_module, in_channel=128, depth=256, num_units=num_layers[2])
        block4 = get_block(unit_module, in_channel=256, depth=512, num_units=num_layers[3])
        self.body = nn.Sequential(block1, block2, block3, block4)

        self.bn2 = nn.BatchNorm2d(512, eps=1e-05)
        self.dropout = nn.Dropout(p=self.dropout_ratio, inplace=True)
        self.fc = nn.Linear(512 * (input_size // 16) ** 2, 512)
        self.features = nn.BatchNorm1d(512, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False
        self.amp = amp

        self._initialize_weights()

    def forward(self, x):
        with torch.cuda.amp.autocast(self.amp):
            x = self.input_layer(x)
            x = self.body(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)
        x = self.fc(x.float() if self.amp else x)
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()


def IR_50(input_size, **kwargs):
    """Constructs a ir-50 model.
    """
    model = IResNet(input_size, [3, 4, 14, 3], 'ir', **kwargs)

    return model
