"""
Contain some self-contained modules.
"""
import torch
import torch.nn as nn
import numpy as np
import json
import os


class LayerNorm(nn.Module):
    """
    Simple 1D LayerNorm.
    """

    def __init__(self, features, center=True, scale=False, eps=1e-6):
        super().__init__()
        self.center = center
        self.scale = scale
        self.eps = eps
        if self.scale:
            self.scale_param = nn.Parameter(torch.ones(features))
        else:
            self.scale_param = None
        if self.center:
            self.center_param = nn.Parameter(torch.zeros(features))
        else:
            self.center_param = None

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        output = (x - mean) / (std + self.eps)
        if self.scale:
            output = output * self.scale_param
        if self.center:
            output = output + self.center_param
        return output


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def zeros(*sizes, torch_device=None, **kwargs):
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    return torch.ones(*sizes, **kwargs, device=torch_device)


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def update_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def convert_to_dict(obj):
    dicts = {}
    dicts.update(obj.__dict__)
    return dicts


class SaveJson(object):
    def save_file(self, path, obj):
        item = convert_to_dict(obj)
        item['device'] = None
        item['env'] = None
        item = json.dumps(item, ensure_ascii=False, indent=4, separators=(',', ':'))
        try:
            if not os.path.exists(path):
                os.makedirs(path)
                with open(f'{path}/rl_config.json', "w", encoding='utf-8') as f:
                    f.write(item + "\n", )
            else:
                with open(path, "a", encoding='utf-8') as f:
                    f.write(item + "\n")
        except Exception as e:
            print("write error==>", e)



