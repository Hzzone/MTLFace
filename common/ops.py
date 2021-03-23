import torch
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from typing import Union
import os.path as osp
import os
import time
from torchvision.utils import save_image
import torch.distributed as dist
import math
import inspect
from torch._six import container_abcs, string_classes
import warnings


def group2onehot(groups, age_group):
    code = torch.eye(age_group)[groups.squeeze()]
    if len(code.size()) > 1:
        return code
    return code.unsqueeze(0)


def group2feature(group, age_group, feature_size):
    onehot = group2onehot(group, age_group)
    return onehot.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, feature_size, feature_size)


def get_norm_layer(norm_layer, module, **kwargs):
    if norm_layer == 'none':
        return module
    elif norm_layer == 'bn':
        return nn.Sequential(
            module,
            nn.BatchNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'in':
        return nn.Sequential(
            module,
            nn.InstanceNorm2d(module.out_channels, **kwargs)
        )
    elif norm_layer == 'sn':
        return nn.utils.spectral_norm(module, **kwargs)
    else:
        return NotImplementedError


def get_varname(var):
    """
    Gets the name of var. Does it from the out most frame inner-wards.
    :param var: variable to get name from.
    :return: string
    """
    for fi in reversed(inspect.stack()):
        names = [var_name for var_name, var_val in fi.frame.f_locals.items() if var_val is var]
        if len(names) > 0:
            return names[0]


def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k.replace('module.', '')  # remove `module.`
        new_state_dict[namekey] = v
    return new_state_dict


class LoggerX(object):

    def __init__(self, save_root):
        assert dist.is_initialized()
        self.models_save_dir = osp.join(save_root, 'save_models')
        self.images_save_dir = osp.join(save_root, 'save_images')
        os.makedirs(self.models_save_dir, exist_ok=True)
        os.makedirs(self.images_save_dir, exist_ok=True)
        self._modules = []
        self._module_names = []
        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_rank()

    @property
    def modules(self):
        return self._modules

    @property
    def module_names(self):
        return self._module_names

    @modules.setter
    def modules(self, modules):
        for i in range(len(modules)):
            self._modules.append(modules[i])
            self._module_names.append(get_varname(modules[i]))

    def checkpoints(self, epoch):
        if self.local_rank != 0:
            return
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            torch.save(module.state_dict(), osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch)))

    def load_checkpoints(self, epoch):
        for i in range(len(self.modules)):
            module_name = self.module_names[i]
            module = self.modules[i]
            module.load_state_dict(load_network(osp.join(self.models_save_dir, '{}-{}'.format(module_name, epoch))))

    def msg(self, stats, step):
        output_str = '[{}] {:05d}, '.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), step)

        for i in range(len(stats)):
            if isinstance(stats, (list, tuple)):
                var = stats[i]
                var_name = get_varname(stats[i])
            elif isinstance(stats, dict):
                var_name, var = list(stats.items())[i]
            else:
                raise NotImplementedError
            if isinstance(var, torch.Tensor):
                var = var.detach().mean()
                var = reduce_tensor(var)
                var = var.item()
            output_str += '{} {:2.5f}, '.format(var_name, var)

        if self.local_rank == 0:
            print(output_str)

    def save_image(self, grid_img, n_iter, sample_type):
        save_image(grid_img, osp.join(self.images_save_dir,
                                      '{}_{}_{}.jpg'.format(n_iter, self.local_rank, sample_type)),
                   nrow=1)


def reduce_loss(*loss):
    return [reduce_tensor(l.detach().mean()).item() for l in loss]


def reduce_tensor(rt):
    rt = rt.clone()
    if dist.is_initialized():
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1
    rt /= world_size
    return rt


def convert_to_ddp(*modules):
    modules = [x.cuda() for x in modules]
    if dist.is_initialized():
        rank = dist.get_rank()
        modules = [torch.nn.parallel.DistributedDataParallel(x,
                                                             device_ids=[rank, ],
                                                             output_device=rank) for x in modules]

    return modules


def get_dex_age(pred):
    pred = F.softmax(pred, dim=1)
    value = torch.sum(pred * torch.arange(pred.size(1)).to(pred.device), dim=1)
    return value


def apply_weight_decay(*modules, weight_decay_factor=0., wo_bn=True):
    '''
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/5
    Apply weight decay to pytorch model without BN;
    In pytorch:
        if group['weight_decay'] != 0:
            grad = grad.add(p, alpha=group['weight_decay'])
    p is the param;
    :param modules:
    :param weight_decay_factor:
    :return:
    '''
    for module in modules:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.Module.modules
        for m in module.modules():
            if hasattr(m, 'weight'):
                if wo_bn and isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    continue
                if not hasattr(m.weight, 'grad'):
                    warnings.warn('{} has no grad.'.format(m))
                    continue
                m.weight.grad.add_(m.weight, alpha=weight_decay_factor)


def convert_to_cuda(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data.cuda(non_blocking=True)
    elif isinstance(data, container_abcs.Mapping):
        return {key: convert_to_cuda(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, '_fields'):  # namedtuple
        return elem_type(*(convert_to_cuda(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(data, string_classes):
        return [convert_to_cuda(d) for d in data]
    else:
        return data


def age2group(age, age_group):
    if isinstance(age, np.ndarray):
        groups = np.zeros_like(age)
    else:
        groups = torch.zeros_like(age).to(age.device)
    if age_group == 4:
        section = [30, 40, 50]
    elif age_group == 5:
        section = [20, 30, 40, 50]
    elif age_group == 6:
        section = [10, 20, 30, 40, 50]
    elif age_group == 7:
        section = [10, 20, 30, 40, 50, 60]
    elif age_group == 8:
        # 0 - 12, 13 - 18, 19 - 25, 26 - 35, 36 - 45, 46 - 55, 56 - 65, ≥ 66
        section = [12, 18, 25, 35, 45, 55, 65]
    else:
        raise NotImplementedError
    for i, thresh in enumerate(section, 1):
        groups[age > thresh] = i
    return groups
