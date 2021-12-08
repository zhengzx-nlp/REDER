import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from torch.nn import functional as F
import sys
import numpy as np
from scipy import linalg as la


# following example for saving and setting rng here https://pytorch.org/docs/stable/_modules/torch/utils/checkpoint.html
class Deterministic(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cpu_state = None
        self.cuda_in_fwd = None
        self.gpu_devices = None
        self.gpu_states = None

    def record_rng(self, *args):
        self.cpu_state = torch.get_rng_state()
        if torch.cuda._initialized:
            self.cuda_in_fwd = True
            self.gpu_devices, self.gpu_states = get_device_states(*args)

    def forward(self, *args, record_rng=False, set_rng=False, **kwargs):
        if record_rng:
            self.record_rng(*args)

        if not set_rng:
            return self.net(*args, **kwargs)

        rng_devices = []
        if self.cuda_in_fwd:
            rng_devices = self.gpu_devices

        with torch.random.fork_rng(devices=rng_devices, enabled=True):
            torch.set_rng_state(self.cpu_state)
            if self.cuda_in_fwd:
                set_device_states(self.gpu_devices, self.gpu_states)
            return self.net(*args, **kwargs)


# heavily inspired by https://github.com/RobinBruegger/RevTorch/blob/master/revtorch/revtorch.py
# once multi-GPU is confirmed working, refactor and send PR back to source
class ReversibleBlock(nn.Module):
    def __init__(self, f, g, depth=None, send_signal=False):
        super().__init__()
        self.f = Deterministic(f)
        self.g = Deterministic(g)

        self.depth = depth
        self.send_signal = send_signal

    def forward(self, x, f_args={}, g_args={}):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1, y2 = None, None

        if self.send_signal:
            f_args["_reverse"] = g_args["_reverse"] = False
            f_args["_depth"] = g_args["_depth"] = self.depth

        with torch.no_grad():
            y1 = x1 + self.f(x2, record_rng=self.training, **f_args)
            y2 = x2 + self.g(y1, record_rng=self.training, **g_args)

        return torch.cat([y1, y2], dim=2)

    def backward_pass(self, y, dy, f_args={}, g_args={}):
        y1, y2 = torch.chunk(y, 2, dim=2)
        del y

        dy1, dy2 = torch.chunk(dy, 2, dim=2)
        del dy

        if self.send_signal:
            f_args["_reverse"] = g_args["_reverse"] = True
            f_args["_depth"] = g_args["_depth"] = self.depth

        with torch.enable_grad():
            y1.requires_grad = True
            gy1 = self.g(y1, set_rng=True, **g_args)
            torch.autograd.backward(gy1, dy2)

        with torch.no_grad():
            x2 = y2 - gy1
            del y2, gy1

            dx1 = dy1 + y1.grad
            del dy1
            y1.grad = None

        with torch.enable_grad():
            x2.requires_grad = True
            fx2 = self.f(x2, set_rng=True, **f_args)
            torch.autograd.backward(fx2, dx1, retain_graph=True)

        with torch.no_grad():
            x1 = y1 - fx2
            del y1, fx2

            dx2 = dy2 + x2.grad
            del dy2
            x2.grad = None

            x = torch.cat([x1, x2.detach()], dim=2)
            dx = torch.cat([dx1, dx2], dim=2)

        return x, dx


class IrreversibleBlock(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=2)
        y1 = x1 + self.f(x2, **f_args)
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=2)


class _ReversibleFunction(Function):
    @staticmethod
    def forward(ctx, x, blocks, kwargs):
        ctx.kwargs = kwargs
        for block in blocks:
            x = block(x, **kwargs)
        ctx.y = x.detach()
        ctx.blocks = blocks
        return x

    @staticmethod
    def backward(ctx, dy):
        y = ctx.y
        kwargs = ctx.kwargs
        for block in ctx.blocks[::-1]:
            y, dy = block.backward_pass(y, dy, **kwargs)
        return dy, None, None


class ReversibleSequence(nn.Module):
    def __init__(self, blocks, layer_dropout=0.0, reverse_thres=0, send_signal=False):
        super().__init__()
        self.layer_dropout = layer_dropout
        self.reverse_thres = reverse_thres

        self.blocks = nn.ModuleList(
            [ReversibleBlock(f, g, depth, send_signal) for depth, (f, g) in enumerate(blocks)]
        )
        self.irrev_blocks = nn.ModuleList([IrreversibleBlock(f=f, g=g) for f, g in blocks])

    def forward(self, x, arg_route=(True, True), **kwargs):
        reverse = x.shape[1] > self.reverse_thres
        blocks = self.blocks if reverse else self.irrev_blocks

        if self.training and self.layer_dropout > 0:
            to_drop = torch.empty(len(self.blocks)).uniform_(0, 1) < self.layer_dropout
            blocks = [block for block, drop in zip(self.blocks, to_drop) if not drop]
            blocks = self.blocks[:1] if len(blocks) == 0 else blocks

        f_args, g_args = map(lambda route: kwargs if route else {}, arg_route)
        block_kwargs = {"f_args": f_args, "g_args": g_args}

        if not reverse:
            for block in blocks:
                x = block(x, **block_kwargs)
            return x

        return _ReversibleFunction.apply(x, blocks, block_kwargs)


class ReversibleBlockNew(nn.Module):
    def __init__(self, f, g):
        super().__init__()
        self.f = f
        self.g = g
        self.f_seed, self.g_seed = None, None

    def forward(self, x, f_args, g_args):
        x1, x2 = torch.chunk(x, 2, dim=2)

        if self.training:
            self._init_seed("f_seed")
        y1 = x1 + self.f(x2, **f_args)

        if self.training:
            self._init_seed("g_seed")
        y2 = x2 + self.g(y1, **g_args)
        return torch.cat([y1, y2], dim=2)

    def reverse(self, y, f_args, g_args, requires_grad=False):
        y1, y2 = torch.chunk(y, 2, dim=2)
        with torch.set_grad_enabled(requires_grad):
            # set seed to have correct dropout
            """ X_2 = Y_2 - g(Y_1) """
            torch.manual_seed(self.g_seed)
            x2 = y2 - self.g(y1)

            # set seed to have correct dropout
            """ X_1 = Y_1 - f(X_2) """
            torch.manual_seed(self.f_seed)
            x1 = y1 - self.f(x2)

        x = torch.cat([x1, x2], dim=2)
        return x

    def _init_seed(self, seed_name):
        """
        This function sets a new seed for the attention layer to make dropout deterministic for both forward calls: 1
        normal forward call and 1 forward call in backward to recalculate activations.
        """

        # randomize seeds
        # use cuda generator if available
        if hasattr(torch.cuda, "default_generators") and len(torch.cuda.default_generators) > 0:
            # GPU
            device_idx = torch.cuda.current_device()
            seed = torch.cuda.default_generators[device_idx].seed()
        else:
            # CPU
            seed = int(torch.seed() % sys.maxsize)
        setattr(self, seed_name)
        torch.manual_seed(seed)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class InvertibleLinear(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        weight = torch.qr(torch.randn(in_features, in_features))[0].contiguous()
        self.weight = nn.Parameter(weight)

        self.register_buffer("inv_weight", torch.inverse(self.weight))

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}".format(self.in_features, self.in_features)

    def _get_weight(self, reverse=False):
        if reverse:
            if self.training:
                _weight = self.weight.detach().float()
                self.inv_weight = torch.inverse(_weight).to(self.weight)
            return self.inv_weight
        return self.weight

    def forward(self, input, reverse=False):
        weight = self._get_weight(reverse=reverse)
        output = F.linear(input, weight)
        return output

    def reverse(self, output):
        return self.forward(output, reverse=True)





def logabs(x): return torch.log(torch.abs(x))


class InvConv1dLU(nn.Module):
    def __init__(self, in_seq):
        super().__init__()

        weight = np.random.randn(in_seq, in_seq)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer('w_p', w_p)
        self.register_buffer('u_mask', torch.from_numpy(u_mask))
        self.register_buffer('l_mask', torch.from_numpy(l_mask))
        self.register_buffer('s_sign', torch.sign(w_s))
        self.register_buffer('l_eye', torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l.contiguous())
        self.w_s = nn.Parameter(logabs(w_s).contiguous())
        self.w_u = nn.Parameter(w_u.contiguous())

    def forward(self, input):
        # input: (bsz, L, d)
        # input: (bsz, d, L)

        weight = self.calc_weight()
        out = F.linear(input.transpose(2, 1), weight).transpose(2, 1)

        return out

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight
    
    def reverse(self, output):
        weight = self.calc_weight()

        return F.linear(output.transpose(2, 1), weight.inverse()).transpose(2, 1)


def swap_halves(feature):
    # feature: [b, l, d+d]
    half1, half2 = torch.chunk(feature, chunks=2, dim=-1)
    return torch.cat([half2, half1], dim=-1)


