import torch
# import transformers
from torch import nn
from functools import partial

from .linear import forward as linear_forward
from .conv2d import forward as conv2d_forward

supports = {
    nn.Linear: linear_forward,
    nn.Conv2d: conv2d_forward,
}

class ISHReshaper(object):
    def __init__(self, strategy, param):
        self.param = param
        self.reserve = 1 - param

        self.select = getattr(self, f"cache_{strategy}")
        self.pad = getattr(self, f"load_{strategy}")

    def cache_minksample_expscale(self, x: torch.Tensor, ctx=None):
        shape = x.shape
        x = x.reshape(shape[0], -1)
        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1])
        
        x, idxs = x.abs().topk(int(x.shape[1] * self.reserve), dim=1, sorted=False)
        x.dropped = True # provide a flag for act judges

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1])

        # apply sharpening
        scale = s1 / s2
        x = x * torch.exp(scale[:, None])

        ctx.idxs = idxs
        ctx.shape = shape
        return x
    
    def load_minksample_expscale(self, x, ctx=None):      
        print(ctx.idxs.shape, x.shape, )
        return torch.zeros(
            ctx.shape, device=x.device, dtype=x.dtype
        ).scatter_(1, ctx.idxs, x)

    def cache_expscale(self, x: torch.Tensor, ctx=None):
        input = x.clone()
        shape = x.shape
        x = x.reshape(shape[0], -1)
        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1])
        
        x, idxs = x.abs().topk(int(x.shape[1] * self.reserve), dim=1, sorted=False)
        x.dropped = True # provide a flag for act judges

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1])

        # apply sharpening
        scale = s1 / s2

        if len(shape) == 4:
            input = input * torch.exp(scale[:, None, None, None])
        elif len(shape) == 2:
            input = input * torch.exp(scale[:, None])
        else:
            raise NotImplementedError

        ctx.idxs = idxs
        ctx.shape = shape
        return input
    
    def load_expscale(self, x, ctx=None):      
        return x


    def cache_minksample_lnscale(self, x: torch.Tensor, ctx=None):
        shape = x.shape
        x = x.reshape(shape[0], -1)
        # calculate the sum of the input per sample
        s1 = x.sum(dim=[1])
        
        x, idxs = x.abs().topk(int(x.shape[1] * self.reserve), dim=1, sorted=False)
        x.dropped = True # provide a flag for act judges

        # calculate new sum of the input per sample after pruning
        s2 = x.sum(dim=[1])

        # apply sharpening
        scale = s1 / s2
        x = x * scale[:, None]

        ctx.idxs = idxs
        ctx.shape = shape
        return x
    
    def load_minksample_lnscale(self, x, ctx=None):      
        return torch.zeros(
            ctx.shape, device=x.device, dtype=x.dtype
        ).scatter_(1, ctx.idxs, x)

    @staticmethod
    def transfer(model, strategy, gamma, autocast):
        _type = type(model)
        if _type in supports and not hasattr(model, 'no_ish'):
            ish_reshaper = ISHReshaper(strategy, gamma)
            ish_reshaper.autocast = autocast # just for recording
            model.forward = partial(supports[_type], model)
            model.ish_reshaper = ish_reshaper
            print(f"{_type}.forward => ish.{strategy}.{_type}.forward")
        for child in model.children():
            ISHReshaper.transfer(child, strategy, gamma, autocast)
        return model

    
def to_ish(model: nn.Module, strategy: str, param: float, autocast: bool = False, layer = None):
    if layer == "r1":
        if hasattr(model, 'module'):
            ISHReshaper.transfer(model.module.fc, strategy, param, autocast)
        else:
            ISHReshaper.transfer(model.fc, strategy, param, autocast)

    elif layer == "all":
        ISHReshaper.transfer(model, strategy, param, autocast)

    return model
