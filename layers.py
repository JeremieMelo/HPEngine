#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-12 18:30:27
@LastEditTime: 2019-07-28 00:32:04
'''
# from scipy.linalg import circulant
import math
import random
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch.nn import init

# from matrix_parametrization import *
from utils import *


class ReLUN(nn.Hardtanh):
    r"""Applies the element-wise function:

    .. math::
        \text{ReLUN}(x) = \min(\max(0,x), N)

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: scripts/activation_images/ReLU6.png

    Examples::

        >>> m = nn.ReLUN(N)
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, N, inplace=False):
        super(ReLUN, self).__init__(0., N, inplace)

    def extra_repr(self):
        inplace_str = 'inplace' if self.inplace else ''
        return inplace_str


class MZILinear(nn.Module):
    def __init__(self, in_channel, out_channel, use_bias=None, device=torch.device("cuda")):
        super(MZILinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.device = device
        self.phi_list_U = None
        self.delta_list_U = None
        self.phi_list_V = None
        self.delta_list_V = None
        self.sigma_list = None
        self.U = None
        self.Sigma = None
        self.V = None
        self.W = None
        self.decomposer = RealUniaryDecomposerPyTorch(
            device=self.device, timer=False, use_multithread=False, n_thread=12)

        self.use_bias = use_bias
        if(use_bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)

        self.buildPhi()
        self.buildDelta()
        self.buildSigma()

    def buildPhi(self):
        self.phi_list_V = Parameter(torch.Tensor(
            self.in_channel*(self.in_channel-1)//2).to(self.device))
        self.phi_list_U = Parameter(torch.Tensor(
            self.out_channel*(self.out_channel-1)//2).to(self.device))
        self.register_parameter("phi_U", self.phi_list_U)
        self.register_parameter("phi_V", self.phi_list_V)

    def buildDelta(self):
        self.delta_list_V = Parameter(
            torch.Tensor(self.in_channel).to(self.device))
        self.delta_list_U = Parameter(
            torch.Tensor(self.out_channel).to(self.device))
        self.register_parameter("delta_U", self.delta_list_U)
        self.register_parameter("delta_V", self.delta_list_V)

    def buildSigma(self):
        M, N = self.out_channel, self.in_channel
        self.sigma_list = Parameter(torch.Tensor(min(M, N)).to(self.device))
        Sigma = torch.diag(self.sigma_list).to(self.device)
        if(M > N):
            self.Sigma = torch.cat(
                [Sigma, torch.zeros(M - N, N).to(self.device)], dim=0).to(torch.float32)
        elif(M < N):
            self.Sigma = torch.cat(
                [Sigma, torch.zeros(M, N - M).to(self.device)], dim=1).to(torch.float32)
        else:
            self.Sigma = Sigma
        self.register_parameter("sigma_list", self.sigma_list)

    def init_weights(self):
        init.normal_(self.phi_list_U)
        init.normal_(self.phi_list_V)
        init.normal_(self.delta_list_U)
        init.normal_(self.delta_list_V)
        init.normal_(self.sigma_list)

        if self.use_bias is not None:
            init.uniform_(self.bias, 0, 0)

    def forward(self, x):
        self.U = self.decomposer.reconstruct_2(
            self.delta_list_U, self.phi_list_U)
        self.V = self.decomposer.reconstruct_2(
            self.delta_list_V, self.phi_list_V)
        self.W = torch.mm(self.U, torch.mm(self.Sigma, self.V))

        out = F.linear(x, self.W, bias=self.bias)
        return out


class USVLinear(nn.Module):
    def __init__(self, in_channel, out_channel, use_bias=False, S_trainable=True, device=torch.device("cuda")):
        super(USVLinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.S_trainable = S_trainable
        self.device = device
        self.U = None
        self.S = None
        self.V = None
        self.W = None
        self.Sigma = None

        self.use_bias = use_bias
        if(use_bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)

        self.buildUSV()

    def buildUSV(self):
        self.U = Parameter(torch.Tensor(self.out_channel, self.out_channel).to(
            self.device).to(torch.float32))
        self.S = Parameter(torch.Tensor(min(self.out_channel, self.in_channel)).to(
            self.device).to(torch.float32), requires_grad=self.S_trainable)

        self.V = Parameter(torch.Tensor(self.in_channel, self.in_channel).to(
            self.device).to(torch.float32))
        self.register_parameter("U", self.U)
        self.register_parameter("V", self.V)
        self.register_parameter("S", self.S)

    def buildSigma(self, S):
        M, N = self.out_channel, self.in_channel
        Sigma = torch.diag(S).to(self.device).to(torch.float32)
        if(M > N):
            Sigma = torch.cat(
                [Sigma, torch.zeros(M - N, N).to(self.device).to(torch.float32)], dim=0)
        elif(M < N):
            Sigma = torch.cat(
                [Sigma, torch.zeros(M, N - M).to(self.device).to(torch.float32)], dim=1)

        return Sigma

    def init_weights(self):
        init.orthogonal_(self.U)
        init.orthogonal_(self.V)
        init.ones_(self.S)

        if self.use_bias == True:
            init.uniform_(self.bias, 0, 0)
        if(self.S_trainable == False):
            self.Sigma = self.buildSigma(self.S)

    def buildWeight_from_USV(self, U, S, V):
        Sigma = self.buildSigma(S) if self.S_trainable == True else self.Sigma
        W = torch.mm(U, torch.mm(Sigma, V))
        return W

    def forward(self, x):
        self.W = self.buildWeight_from_USV(self.U, self.S, self.V)
        out = F.linear(x, self.W, bias=self.bias)

        return out


class BlockUSVLinear(nn.Module):
    def __init__(self, in_channel, out_channel, block_size, use_bias=False, S_trainable=True, device=torch.device("cuda")):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.block_size = block_size
        self.in_block = in_channel // block_size
        self.out_block = out_channel // block_size
        assert in_channel % block_size == 0 and out_channel % block_size == 0, f"[E] block size {block_size} must be common divisor of in channel {in_channel} and out channel {out_channel}"
        self.S_trainable = S_trainable
        self.device = device
        self.U = None
        self.S = None
        self.V = None
        self.W = None
        self.Sigma = None

        self.use_bias = use_bias
        if(use_bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)

        self.buildUSV()

    def buildUSV(self):

        self.U = Parameter(torch.Tensor(self.out_block, self.in_block,
                                        self.block_size, self.block_size).to(self.device).to(torch.float32))
        self.S = Parameter(torch.Tensor(self.out_block, self.in_block, self.block_size, 1).to(
            self.device).to(torch.float32), requires_grad=self.S_trainable)

        self.V = Parameter(torch.Tensor(self.out_block, self.in_block,
                                        self.block_size, self.block_size).to(self.device).to(torch.float32))
        self.register_parameter("U", self.U)
        self.register_parameter("V", self.V)
        self.register_parameter("S", self.S)

    def init_weights(self):
        for i in range(self.out_block):
            for j in range(self.in_block):
                init.orthogonal_(self.U[i, j, ...])
                init.orthogonal_(self.V[i, j, ...])
        init.ones_(self.S)

        if self.use_bias == True:
            init.uniform_(self.bias, 0, 0)

    def buildWeight_from_USV(self, U, S, V):
        W = torch.matmul(U, S * V)
        return W

    def forward(self, x, U=None, V=None):
        if(U is None):
            U = self.U
        if(V is None):
            V = self.V
        self.W = self.buildWeight_from_USV(U, self.S, V)

        W = merge_chunks(self.W)

        out = F.linear(x, W, bias=self.bias)
        return out


class ConditionalQuantizationWithGammaNoiseOfUnitary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, v_bit, v_pi, v_max, lambda3, gamma_noise_std, decay_rate, learning_rate, clip_voltage, lower_thres, upper_thres, prune_mask, output_device):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input, prune_mask)

        output_q, output_qn = conditional_update_voltage_of_unitary_cpu(
            W=input,
            v_bit=v_bit,
            v_pi=v_pi,
            v_max=v_max,
            lambda3=lambda3,
            voltage_mask=None,
            voltage_backup=None,
            gamma_noise_std=gamma_noise_std,
            weight_decay_rate=decay_rate,
            learning_rate=learning_rate,
            clip_voltage=clip_voltage,
            lower_thres=lower_thres,
            upper_thres=upper_thres,
            return_ori=True,
            output_device=output_device)
        output_q[prune_mask, :, :], output_qn[prune_mask, :, :] = 0, 0
        return output_q.data, output_qn

    @staticmethod
    def backward(ctx, grad_output_q, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, prune_mask = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[prune_mask, :, :] = 0
        mean, sigma = grad_input.mean(), grad_input.std()
        grad_input = (grad_input - mean).clamp(-3*sigma, 3*sigma) + mean
        return grad_input, None, None, None, None, None, None, None, None, None, None, None, None


class CirculantLinear(nn.Module):
    def __init__(self, in_channel, out_channel, mini_block=4, use_bias=None, device=torch.device("cuda")):
        super(CirculantLinear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mini_block = mini_block
        self.device = device

        self.eigens = None
        self.circs = None
        self.use_bias = use_bias
        assert in_channel % mini_block == 0 and out_channel % mini_block == 0, "in_channel and out_channel must be multiples of mini block size"
        self.gridDim_y = out_channel // mini_block
        self.gridDim_x = in_channel // mini_block

        if(use_bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)
        self.mask = []

        self.buildEigens()
        # self.init_weights()
        # self.buildCirculantMatrices()

    def buildEigens(self):
        # self.eigens = [[Parameter(torch.Tensor(self.mini_block).to(self.device)) for j in range(
        #     self.in_channel//self.mini_block)] for i in range(self.out_channel//self.mini_block)]
        # self.eigens_flatten = nn.ParameterList(list(chain.from_iterable(self.eigens)))
        self.eigens = [Parameter(torch.Tensor(self.mini_block).to(
            self.device)) for i in range(self.gridDim_x * self.gridDim_y)]
        self.eigens = nn.ParameterList(self.eigens)
        print(len(self.eigens))
        # print(self.eigens)

    def init_weights(self):
        for i in range(self.gridDim_x * self.gridDim_y):
            init_stddev = np.sqrt(1. / self.mini_block)
            init.normal_(self.eigens[i], std=init_stddev)

        if self.use_bias is not None:
            init.uniform_(self.bias, 0, 0)

    def buildCirculantMatrices(self):

        self.circs = [[circulant(self.eigens[i * self.gridDim_x + j])
                       for j in range(self.gridDim_x)] for i in range(self.gridDim_y)]
        # print(self.circs)

    def getGroupLassoRegularizer(self, threshold=0, l1_fraction=0):
        self.threshold = threshold
        self.l1_fraction = l1_fraction
        self.circ_lives = torch.ByteTensor(
            self.gridDim_x * self.gridDim_y).to(self.device)
        ### l2_norm ###
        self.l2_norm = torch.Tensor(
            self.gridDim_x * self.gridDim_y).to(self.device)
        for i in range(self.gridDim_x * self.gridDim_y):
            self.l2_norm[i] = torch.norm(self.eigens[i], p=2) / self.mini_block

        ### l1_norm ###
        if(self.l1_fraction > 0.0):
            self.l1_norm = torch.Tensor(
                self.gridDim_x * self.gridDim_y).to(self.device)
            for i in range(self.gridDim_x * self.gridDim_y):
                self.l1_norm[i] = torch.norm(
                    self.eigens[i], p=1) / self.mini_block
            self.norm = l1_fraction * self.l1_norm + \
                (1.0 - l1_fraction) * self.l2_norm
        else:
            self.norm = self.l2_norm

        self.circ_lives = self.norm > threshold
        self.reg_loss = torch.mean(self.norm)
        return self.reg_loss

    def createMask(self, drop_rate=0.2, mask=None):
        assert 0 <= drop_rate < 1, "[E] Drop rate must be in the range [0,1)"
        if(mask == None):
            n_eigens = len(self.eigens)
            mask = random.sample(range(n_eigens), k=int(n_eigens * drop_rate))
        self.mask = set(mask)
        print(f"mask len: {len(self.mask)} : {self.mask}")

    def forward(self, x):
        # outputs = [[F.linear(x[:, j*self.mini_block:(j+1)*self.mini_block], self.circs[i][j], None)
        #             for j in range(self.gridDim_x)] for i in range(self.gridDim_y)]
        if(len(self.mask) > 0):
            zero_tensor = torch.zeros(
                [x.size(0), self.mini_block]).to(self.device)
            outputs = [[zero_tensor if i * self.gridDim_x + j in self.mask else circulant_multiply(
                self.eigens[i * self.gridDim_x + j], x[:, j * self.mini_block:(j + 1) * self.mini_block]) for j in range(self.gridDim_x)] for i in range(self.gridDim_y)]
        else:
            outputs = [[circulant_multiply(self.eigens[i * self.gridDim_x + j], x[:, j * self.mini_block:(
                j + 1) * self.mini_block]) for j in range(self.gridDim_x)] for i in range(self.gridDim_y)]
        outputs = [torch.sum(torch.stack(outputs[i], dim=2), dim=2)
                   for i in range(self.gridDim_y)]
        outputs = torch.cat(outputs, dim=1)
        if(self.use_bias):
            outputs = outputs + self.bias

        # outputs = self.act(outputs)
        # print(outputs, outputs.size())
        return outputs


class CirculantLinear_v2(nn.Module):
    '''Random structured sparsity by random drop mask'''

    def __init__(self, in_channel, out_channel, mini_block=4, use_bias=None, group_lasso=False, device=torch.device("cuda")):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mini_block = mini_block
        self.device = device

        self.eigens = None
        self.circs = None
        self.use_bias = use_bias
        self.group_lasso = group_lasso
        assert in_channel % mini_block == 0 and out_channel % mini_block == 0, "in_channel and out_channel must be multiples of mini block size"
        self.gridDim_y = out_channel // mini_block
        self.gridDim_x = in_channel // mini_block

        if(use_bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)
        self.mask = None

        self.buildEigens()

    def buildEigens(self):
        self.eigens = Parameter(torch.Tensor(
            self.gridDim_y, self.gridDim_x, self.mini_block).to(self.device))
        self.register_parameter("weight", self.eigens)

    def init_weights(self):
        init.kaiming_normal_(self.eigens)

        if self.use_bias is not None:
            init.uniform_(self.bias, 0, 0)

    def getGroupLassoRegularizer(self, threshold=0, l1_fraction=0):
        self.threshold = threshold
        self.l1_fraction = l1_fraction
        self.circ_lives = torch.ByteTensor(
            self.gridDim_x * self.gridDim_y).to(self.device)
        ### l2_norm ###
        self.l2_norm = torch.Tensor(
            self.gridDim_x * self.gridDim_y).to(self.device)
        for i in range(self.gridDim_x * self.gridDim_y):
            self.l2_norm[i] = torch.norm(self.eigens[i], p=2) / self.mini_block

        ### l1_norm ###
        if(self.l1_fraction > 0.0):
            self.l1_norm = torch.Tensor(
                self.gridDim_x * self.gridDim_y).to(self.device)
            for i in range(self.gridDim_x * self.gridDim_y):
                self.l1_norm[i] = torch.norm(
                    self.eigens[i], p=1) / self.mini_block
            self.norm = l1_fraction * self.l1_norm + \
                (1.0 - l1_fraction) * self.l2_norm
        else:
            self.norm = self.l2_norm

        self.circ_lives = self.norm > threshold
        self.reg_loss = torch.mean(self.norm)
        return self.reg_loss

    def createMask(self, drop_rate=0.1, mask=None):
        assert 0 <= drop_rate < 1, "[E] Drop rate must be in the range [0,1)"
        if(mask == None):
            n_eigens = self.gridDim_x * self.gridDim_y
            mask = sorted(random.sample(
                range(n_eigens), k=int(n_eigens * drop_rate)))
        else:
            mask = list(set(mask))
        # print(f"mask len: {len(self.mask)} : {self.mask}")
        if(len(mask) > 0):
            self.mask = torch.ones(
                self.gridDim_y * self.gridDim_x).to(self.device)
            zero_idx = torch.LongTensor(mask).to(self.device)
            self.mask.scatter_(0, zero_idx, 0)
            self.mask = self.mask.contiguous().view(self.gridDim_y, self.gridDim_x, 1)
        else:
            self.mask = None

    def forward(self, x):
        # print(f"eigens: {self.eigens.size()}")
        # f_eigens = torch.rfft(self.eigens, 1, normalized=True, onesided=False)
        f_eigens = torch.rfft(self.eigens, 1, onesided=False)
        # print(f"f_eigens: {f_eigens.size()}")
        x = x.contiguous().view([-1, self.gridDim_x, self.mini_block])
        # print(f"x: {x.size()}")
        # f_x = torch.rfft(x, 1, normalized=True, onesided=False)
        f_x = torch.rfft(x, 1, onesided=False)
        # print(f"f_x: {f_x.size()}")
        f_x_duplicate = f_x.unsqueeze(1).repeat(
            1, self.gridDim_y, 1, 1, 1)  # [batch, "p", q, k, 2]
        # print(f"f_x_duplicate: {f_x_duplicate.size()}")
        f_element_mul = complex_mult(f_eigens, f_x_duplicate)
        # print(f"f_ele: {f_element_mul.size()}")
        # if_element_mul = torch.irfft(f_element_mul, 1, normalized=True, onesided=False, signal_sizes=(self.mini_block, ))
        if_element_mul = torch.irfft(
            f_element_mul, 1, onesided=False, signal_sizes=(self.mini_block, ))
        # print(f"if_ele: {if_element_mul.size()}")
        if(self.mask is not None):
            if_element_mul = if_element_mul * self.mask
            # print(f"mask: {self.mask.size()}")
            # print(f"if_ele: {if_element_mul.size()}")
        outputs = torch.sum(if_element_mul, dim=2)
        # print(f"outputs: {outputs.size()}")
        outputs = outputs.view([-1, self.gridDim_y * self.mini_block])
        # print(f"outputs: {outputs.size()}")

        if(self.use_bias):
            outputs = outputs + self.bias

        return outputs


class CirculantLinear_v3(nn.Module):
    '''Add group lasso sparsity and drop mask'''

    def __init__(self, in_channel, out_channel, mini_block=4, group_lasso=False, use_bias=None, device=torch.device("cuda")):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mini_block = mini_block
        self.device = device

        self.eigens = None
        self.circs = None
        self.use_bias = use_bias
        self.group_lasso = group_lasso
        self.lasso = None
        assert in_channel % mini_block == 0 and out_channel % mini_block == 0, "in_channel and out_channel must be multiples of mini block size"
        self.gridDim_y = out_channel // mini_block
        self.gridDim_x = in_channel // mini_block

        if(use_bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)
        self.mask = None

        self.buildEigens()

    def buildEigens(self):
        self.eigens = Parameter(torch.Tensor(
            self.gridDim_y, self.gridDim_x, self.mini_block).to(self.device))
        self.register_parameter("weight", self.eigens)

    def init_weights(self):
        init.kaiming_normal_(self.eigens)

        if self.use_bias is not None:
            init.uniform_(self.bias, 0, 0)

    def getGroupLasso(self):
        self.lasso = torch.zeros(self.gridDim_y, self.gridDim_x).to(
            self.device) if self.group_lasso == False else torch.norm(self.eigens, p=2, dim=2) / np.sqrt(self.eigens.size(-1))
        return self.lasso

    def forward(self, x):
        # print(f"eigens: {self.eigens.size()}")
        # f_eigens = torch.rfft(self.eigens, 1, normalized=True, onesided=False)
        f_eigens = torch.rfft(self.eigens, 1, onesided=False)
        # print(f"f_eigens: {f_eigens.size()}")
        x = x.contiguous().view([-1, self.gridDim_x, self.mini_block])
        # print(f"x: {x.size()}")
        # f_x = torch.rfft(x, 1, normalized=True, onesided=False)
        f_x = torch.rfft(x, 1, onesided=False)
        # print(f"f_x: {f_x.size()}")
        f_x_duplicate = f_x.unsqueeze(1).repeat(
            1, self.gridDim_y, 1, 1, 1)  # [batch, "p", q, k, 2]
        # print(f"f_x_duplicate: {f_x_duplicate.size()}")
        f_element_mul = complex_mult(f_eigens, f_x_duplicate)
        # print(f"f_ele: {f_element_mul.size()}")
        # if_element_mul = torch.irfft(f_element_mul, 1, normalized=True, onesided=False, signal_sizes=(self.mini_block, ))
        if_element_mul = torch.irfft(
            f_element_mul, 1, onesided=False, signal_sizes=(self.mini_block, ))
        # print(f"if_ele: {if_element_mul.size()}")
        if(self.mask is not None):
            if_element_mul = if_element_mul * self.mask
            # print(f"mask: {self.mask.size()}")
            # print(f"if_ele: {if_element_mul.size()}")
        outputs = torch.sum(if_element_mul, dim=2)
        # outputs = torch.mean(if_element_mul, dim=2)
        # print(f"outputs: {outputs.size()}")
        outputs = outputs.view([-1, self.gridDim_y * self.mini_block])
        # print(f"outputs: {outputs.size()}")

        return outputs


class CirculantLinear_v4(nn.Module):
    '''group lasso sparsity and drop mask
        attenuator normalization -> constraint magnitude = 1 is too strong, relax it to penalty
    '''

    def __init__(self, in_channel, out_channel, mini_block=4, group_lasso=False, use_bias=None, device=torch.device("cuda")):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mini_block = mini_block
        self.device = device

        self.eigens = None
        self.circs = None
        self.use_bias = use_bias
        self.group_lasso = group_lasso
        self.lasso = None
        assert in_channel % mini_block == 0 and out_channel % mini_block == 0, "in_channel and out_channel must be multiples of mini block size"
        self.gridDim_y = out_channel // mini_block
        self.gridDim_x = in_channel // mini_block

        if(use_bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)
        self.mask = None

        self.buildEigens()

    def buildEigens(self):
        '''Directly train fft(w), not w. Don't care about w, only care about fft(w)'''
        self.eigens = Parameter(torch.Tensor(
            self.gridDim_y, self.gridDim_x, self.mini_block).to(self.device))
        self.register_parameter("weight", self.eigens)

    def init_weights(self):
        init.kaiming_normal_(self.eigens)
        self.eigens.data = self.eigens.data  # +0.12
        # f_eigens = torch.Tensor(self.gridDim_y, self.gridDim_x, self.mini_block//2+1, 2).to(self.device)
        # init.kaiming_normal_(f_eigens)
        # epsilon = 1e-12
        # f_eigens = F.normalize(torch.clamp(f_eigens, min=np.sqrt(2)*0.5*epsilon), p=2, dim=-1, eps=epsilon)
        # self.eigens.data = torch.irfft(f_eigens, 1, onesided=True, signal_sizes=(self.mini_block,))

        if self.use_bias is not None:
            init.uniform_(self.bias, 0, 0)

    def getGroupLasso(self):
        self.lasso = torch.zeros(self.gridDim_y, self.gridDim_x).to(
            self.device) if self.group_lasso == False else torch.norm(self.eigens, p=2, dim=2) / self.eigens.size(-1)
        return self.lasso

    def getAttenuatorPenalty(self):
        self.attenuator_magnitude = torch.norm(self.f_eigens, p=2, dim=-1)
        self.attenuator_penalty = (self.attenuator_magnitude - 1)**2
        return self.attenuator_penalty

    def forward(self, x):
        # print(f"eigens: {self.eigens.size()}")
        # f_eigens = torch.rfft(self.eigens, 1, normalized=True, onesided=False)
        f_eigens = torch.rfft(self.eigens, 1, onesided=False)
        self.f_eigens = f_eigens
        # print(f"f_eigens: {f_eigens.size()}")
        x = x.contiguous().view([-1, self.gridDim_x, self.mini_block])
        # print(f"x: {x.size()}")
        # f_x = torch.rfft(x, 1, normalized=True, onesided=False)
        f_x = torch.rfft(x, 1, onesided=False)
        # print(f"f_x: {f_x.size()}")
        f_x_duplicate = f_x.unsqueeze(1).repeat(
            1, self.gridDim_y, 1, 1, 1)  # [batch, "p", q, k, 2]
        # print(f"f_x_duplicate: {f_x_duplicate.size()}")
        # f_eigens = F.normalize(F.normalize(f_eigens, p=2, dim=-1), p=2, dim=-1)
        # eplison = 1e-12
        # f_eigens = F.normalize(torch.clamp(f_eigens, min=np.sqrt(2)*0.5*eplison), p=2, dim=-1, eps=eplison)

        f_element_mul = complex_mult(f_eigens, f_x_duplicate)
        # print(f"f_ele: {f_element_mul.size()}")
        # if_element_mul = torch.irfft(f_element_mul, 1, normalized=True, onesided=False, signal_sizes=(self.mini_block, ))
        if_element_mul = torch.irfft(
            f_element_mul, 1, onesided=False, signal_sizes=(self.mini_block, ))
        # print(f"if_ele: {if_element_mul.size()}")
        if(self.mask is not None):
            if_element_mul = if_element_mul * self.mask
            # print(f"mask: {self.mask.size()}")
            # print(f"if_ele: {if_element_mul.size()}")
        outputs = torch.sum(if_element_mul, dim=2)
        # outputs = torch.mean(if_element_mul, dim=2)
        # print(f"outputs: {outputs.size()}")
        outputs = outputs.view([-1, self.gridDim_y * self.mini_block])
        # print(f"outputs: {outputs.size()}")

        return outputs


class CirculantLinear_v5(nn.Module):
    '''group lasso sparsity and drop mask
    '''

    def __init__(self, in_channel, out_channel, mini_block=4, group_lasso=False, use_bias=None, device=torch.device("cuda")):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mini_block = mini_block
        self.device = device

        self.eigens = None
        self.circs = None
        self.use_bias = use_bias
        self.group_lasso = group_lasso
        self.lasso = None
        assert in_channel % mini_block == 0 and out_channel % mini_block == 0, "in_channel and out_channel must be multiples of mini block size"
        self.gridDim_y = out_channel // mini_block
        self.gridDim_x = in_channel // mini_block

        if(use_bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)
        self.mask = None

        self.buildEigens()

    def buildEigens(self):
        '''Directly train fft(w), not w. Don't care about w, only care about fft(w)'''
        self.eigens = Parameter(torch.Tensor(
            self.gridDim_y, self.gridDim_x, self.mini_block).to(self.device))
        self.register_parameter("weight", self.eigens)

    def init_weights(self):
        init.kaiming_normal_(self.eigens)
        self.eigens.data = self.eigens.data  # +0.12

        if self.use_bias is not None:
            init.uniform_(self.bias, 0, 0)

    def getGroupLasso(self):
        self.lasso = torch.zeros(self.gridDim_y, self.gridDim_x).to(
            self.device) if self.group_lasso == False else torch.norm(self.eigens, p=2, dim=2) / self.eigens.size(-1)
        return self.lasso

    def getAttenuatorPenalty(self):
        """Can assume there are amplifiers, don't care about this too much"""
        self.attenuator_magnitude = torch.norm(self.f_eigens, p=2, dim=-1)
        self.attenuator_penalty = (self.attenuator_magnitude - 1)**2
        return self.attenuator_penalty

    def forward(self, x):
        f_eigens = torch.rfft(self.eigens, 1, onesided=False)

        self.f_eigens = f_eigens
        # print(f"f_eigens: {f_eigens.size()}")
        x = x.contiguous().view([-1, self.gridDim_x, self.mini_block])

        # print(f"x: {x.size()}")
        f_x = torch.rfft(x, 1, normalized=True, onesided=False)

        # print(f"f_x: {f_x.size()}")
        f_x_duplicate = f_x.unsqueeze(1).repeat(
            1, self.gridDim_y, 1, 1, 1)  # [batch, "p", q, k, 2]
        # print(f"f_x_duplicate: {f_x_duplicate.size()}")

        f_element_mul = complex_mult(f_eigens, f_x_duplicate)
        # print(f"f_ele: {f_element_mul.size()}")
        if_element_mul = torch.irfft(
            f_element_mul, 1, normalized=True, onesided=False, signal_sizes=(self.mini_block, ))

        # print(f"if_ele: {if_element_mul.size()}")

        outputs = torch.sum(if_element_mul, dim=2)
        # outputs = torch.mean(if_element_mul, dim=2)
        # print(f"outputs: {outputs.size()}")
        outputs = outputs.view([-1, self.gridDim_y * self.mini_block])
        # print(f"outputs: {outputs.size()}")

        return outputs


class CirculantLinear_v6(nn.Module):
    '''group lasso sparsity and drop mask
        attenuator normalization -> constraint magnitude = 1 is too strong, relax it to penalty
        v5: add phase drift noise, requires complex input/output
    '''

    def __init__(self, in_channel, out_channel, mini_block=4, group_lasso=False, phase_drift=False, phase_drift_std=0.05, use_bias=None, device=torch.device("cuda")):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.mini_block = mini_block
        self.device = device

        self.eigens = None
        self.circs = None
        self.use_bias = use_bias
        self.group_lasso = group_lasso
        self.lasso = None
        self.phase_drift = phase_drift
        self.phase_drift_std = phase_drift_std
        assert in_channel % mini_block == 0 and out_channel % mini_block == 0, "in_channel and out_channel must be multiples of mini block size"
        self.gridDim_y = out_channel // mini_block
        self.gridDim_x = in_channel // mini_block

        if(use_bias):
            self.bias = Parameter(torch.Tensor(out_channel).to(self.device))
        else:
            self.register_parameter('bias', None)
        self.mask = None

        self.buildEigens()

    def buildEigens(self):
        '''Directly train fft(w), not w. Don't care about w, only care about fft(w)'''
        self.eigens = Parameter(torch.Tensor(
            self.gridDim_y, self.gridDim_x, self.mini_block).to(self.device))
        self.register_parameter("weight", self.eigens)

    def init_weights(self):
        init.kaiming_normal_(self.eigens)
        self.eigens.data = self.eigens.data  # +0.12
        # f_eigens = torch.Tensor(self.gridDim_y, self.gridDim_x, self.mini_block//2+1, 2).to(self.device)
        # init.kaiming_normal_(f_eigens)
        # epsilon = 1e-12
        # f_eigens = F.normalize(torch.clamp(f_eigens, min=np.sqrt(2)*0.5*epsilon), p=2, dim=-1, eps=epsilon)
        # self.eigens.data = torch.irfft(f_eigens, 1, onesided=True, signal_sizes=(self.mini_block,))

        if self.use_bias is not None:
            init.uniform_(self.bias, 0, 0)

    def getGroupLasso(self):
        self.lasso = torch.zeros(self.gridDim_y, self.gridDim_x).to(
            self.device) if self.group_lasso == False else torch.norm(self.eigens, p=2, dim=2) / self.eigens.size(-1)
        return self.lasso

    def getAttenuatorPenalty(self):
        """Can assume there are amplifiers, don't care about this too much"""
        self.attenuator_magnitude = torch.norm(self.f_eigens, p=2, dim=-1)
        self.attenuator_penalty = (self.attenuator_magnitude - 1)**2
        return self.attenuator_penalty

    def forward(self, x):  # input will be complex number
        # print(f"eigens: {self.eigens.size()}")
        # f_eigens = torch.rfft(self.eigens, 1, normalized=True, onesided=False)
        f_eigens = torch.rfft(self.eigens, 1, onesided=False)
        if(self.phase_drift):
            f_eigens = addPhaseDrift(f_eigens, std=self.phase_drift_std)
        self.f_eigens = f_eigens
        # print(f"f_eigens: {f_eigens.size()}")
        x = x.contiguous().view([-1, self.gridDim_x, self.mini_block, 2])
        if(self.phase_drift):
            x = addPhaseDrift(x, std=self.phase_drift_std)
        # print(f"x: {x.size()}")
        # f_x = torch.rfft(x, 1, normalized=True, onesided=False)
        f_x = torch.fft(x, 1, normalized=True)
        # print(f"f_x: {f_x.size()}")
        f_x_duplicate = f_x.unsqueeze(1).repeat(
            1, self.gridDim_y, 1, 1, 1)  # [batch, "p", q, k, 2]
        # print(f"f_x_duplicate: {f_x_duplicate.size()}")
        # f_eigens = F.normalize(F.normalize(f_eigens, p=2, dim=-1), p=2, dim=-1)
        # eplison = 1e-12
        # f_eigens = F.normalize(torch.clamp(f_eigens, min=np.sqrt(2)*0.5*eplison), p=2, dim=-1, eps=eplison)

        f_element_mul = complex_mult(f_eigens, f_x_duplicate)
        # print(f"f_ele: {f_element_mul.size()}")
        # if_element_mul = torch.irfft(f_element_mul, 1, normalized=True, onesided=False, signal_sizes=(self.mini_block, ))
        if_element_mul = torch.ifft(f_element_mul, 1, normalized=True)
        if(self.phase_drift):
            if_element_mul = addPhaseDrift(
                if_element_mul, std=self.phase_drift_std)
        # print(f"if_ele: {if_element_mul.size()}")

        outputs = torch.sum(if_element_mul, dim=2)
        # outputs = torch.mean(if_element_mul, dim=2)
        # print(f"outputs: {outputs.size()}")
        outputs = outputs.view([-1, self.gridDim_y * self.mini_block, 2])
        # print(f"outputs: {outputs.size()}")

        return outputs


class ModReLU(nn.Module):
    """ A modular ReLU activation function for complex-valued tensors """

    def __init__(self, bias_shape, device=torch.cuda):
        super(ModReLU, self).__init__()
        self.device = device
        if(isinstance(bias_shape, int)):
            self.bias = nn.Parameter(
                torch.Tensor(1, bias_shape).to(self.device))
        else:
            self.bias = nn.Parameter(torch.Tensor(
                1, *bias_shape).to(self.device))
        self.relu = nn.ReLU()
        self.init_bias()

    def init_bias(self):
        init.constant(self.bias, val=0)

    def forward(self, x, eps=1e-5):
        """ ModReLU forward
        Args:
            x (torch.tensor): A torch float tensor with the real and imaginary part
                stored in the last dimension of the tensor; i.e. x.shape = [a, ...,b, 2]
        Kwargs:
            eps (float): A small number added to the norm of the complex tensor for
                numerical stability.
        """
        x_re, x_im = x[..., 0], x[..., 1]
        norm = torch.sqrt(x_re ** 2 + x_im ** 2) + 1e-5
        phase_re, phase_im = x_re / norm, x_im / norm
        activated_norm = self.relu(norm + self.bias)
        modrelu = torch.stack(
            [activated_norm * phase_re, activated_norm * phase_im], -1
        )
        return modrelu



class BiasReLU(nn.Module):
    """ A modular ReLU activation function with bias """

    def __init__(self, bias_shape, max_val=None, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        if(isinstance(bias_shape, int)):
            self.bias = nn.Parameter(
                torch.Tensor(1, bias_shape).to(self.device))
        else:
            self.bias = nn.Parameter(torch.Tensor(
                1, *bias_shape).to(self.device))
        self.max_val = max_val
        self.relu = nn.ReLU() if max_val is None else ReLUN(max_val)
        self.init_bias()

    def init_bias(self):
        init.constant(self.bias, val=0)

    def forward(self, x):
        """ BiasReLU forward
        Args:
            x (torch.tensor): A torch float tensor with the real and imaginary part
                stored in the last dimension of the tensor; i.e. x.shape = [a, ...,b, 2]
        """
        x = self.relu(x + self.bias)
        return x


class DirectionalCoupler2X2(nn.Module):
    def __init__(self, upper_trans=np.sqrt(0.5), device=torch.device("cuda")):
        super().__init__()
        self.upper_trans = upper_trans
        self.device = device
        assert 0 < self.upper_trans < 1, "[E] Upper arm transmission factor must within (0,1)"
        self.lower_trans = np.sqrt(1 - upper_trans**2)
        self.trans_matrix = torch.Tensor([[[self.upper_trans, 0], [0, self.lower_trans]],
                                          [[0, self.lower_trans], [self.upper_trans, 0]]]).to(self.device)

    def forward(self, x1, x2):
        inputs = torch.cat([x1.unsqueeze(0), x2.unsqueeze(0)],
                           dim=0).unsqueeze(0).repeat(2, 1, 1)
        p = complex_mult(self.trans_matrix, inputs)
        p = torch.sum(p, dim=1)
        y1, y2 = p[0, ...], p[1, ...]
        return y1, y2


class DirectionalCoupler2X2_withPhaseShift(nn.Module):
    def __init__(self, upper_trans=np.sqrt(0.5), device=torch.device("cuda")):
        super().__init__()
        self.upper_trans = upper_trans
        self.device = device
        assert 0 < self.upper_trans < 1, "[E] Upper arm transmission factor must within (0,1)"
        self.coupler = DirectionalCoupler2X2(
            upper_trans=self.upper_trans, device=self.device)
        self.phaseshifter270 = PhaseShifter(
            np.pi*1.5, device=self.device)  # -90 degree

    def forward(self, x1, x2):
        x2 = self.phaseshifter270(x2)
        y1, y2 = self.coupler(x1, x2)
        y2 = self.phaseshifter270(y2)
        return y1, y2


class MZI45(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super().__init__()
        self.device = device

    def forward(self, x1, x2):
        y1 = (x1 + x2) / np.sqrt(2)
        y2 = (x2 - x1) / np.sqrt(2)
        return y1, y2


class PhaseShifter(nn.Module):
    def __init__(self, phase, device=torch.device("cuda")):
        super().__init__()
        self.phase = phase
        self.device = device
        self.real = np.cos(self.phase)
        self.imag = np.sin(self.phase)
        self.trans_matrix = torch.Tensor(
            [self.real, self.imag]).to(self.device)

    def forward(self, x):
        y = complex_mult(x, self.trans_matrix)
        return y


class OFFT4(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.coupler = DirectionalCoupler2X2_withPhaseShift(device=self.device)
        self.phaseshifter270 = PhaseShifter(np.pi*1.5, device=self.device)

    def forward(self, x):
        x0, x1, x2, x3 = x[..., 0, :], x[..., 1, :], x[..., 2, :], x[..., 3, :]

        # first stage
        x0, x2 = self.coupler(x0, x2)
        x1, x3 = self.coupler(x1, x3)

        # second stage
        X0, X2 = self.coupler(x0, x1)
        x3 = self.phaseshifter270(x3)
        X1, X3 = self.coupler(x2, x3)

        X = torch.cat([X0.unsqueeze(0), X1.unsqueeze(
            0), X2.unsqueeze(0), X3.unsqueeze(0)], dim=0)

        return X


class OIFFT4(nn.Module):
    def __init__(self, device=torch.device("cuda")):
        super().__init__()
        self.device = device
        self.coupler = DirectionalCoupler2X2_withPhaseShift(device=self.device)
        self.phaseshifter90 = PhaseShifter(np.pi*0.5, device=self.device)

    def forward(self, x):
        x0, x1, x2, x3 = x[..., 0, :], x[..., 1, :], x[..., 2, :], x[..., 3, :]

        # first stage
        x0, x2 = self.coupler(x0, x2)
        x1, x3 = self.coupler(x1, x3)

        # second stage
        X0, X2 = self.coupler(x0, x1)
        x3 = self.phaseshifter90(x3)
        X1, X3 = self.coupler(x2, x3)

        X = torch.cat([X0.unsqueeze(0), X1.unsqueeze(
            0), X2.unsqueeze(0), X3.unsqueeze(0)], dim=0)

        return X


def showOFFT4():
    device = torch.device("cuda")
    offt4 = OFFT4(device=device)
    x = []
    y = []
    count = 0
    plt.figure()
    for x0 in range(2):
        for x1 in range(2):
            for x2 in range(2):
                for x3 in range(2):

                    inp = torch.Tensor(
                        np.array([[x0, 0], [x1, 0], [x2, 0], [x3, 0]])).to(device)
                    X = offt4(inp)
                    # real = X[:,0].cpu().numpy().tolist()
                    # imag = X[:,1].cpu().numpy().tolist()
                    mag = torch.norm(X, p=2, dim=1).cpu().numpy().tolist()
                    X = X.cpu().numpy()
                    freq = (np.arctan2(X[:, 1], X[:, 0])*180/np.pi).tolist()
                    print(count)
                    print(mag)
                    print(freq)
                    print()
                    plt.subplot(4, 4, count+1)
                    plt.bar(freq, mag)
                    count += 1

                    # x.extend(real)
                    # y.extend(imag)
    plt.show()


def uniform_quantize(k, gradient_clip=False):
    class qfn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, input):
            if k == 32:
                out = input
            elif k == 1:
                out = torch.sign(input)
            else:
                n = float(2 ** k - 1)
                out = torch.round(input * n) / n
            return out

        @staticmethod
        def backward(ctx, grad_output):
            grad_input = grad_output.clone()
            if(gradient_clip):
                grad_input.clamp_(-1,1)
            return grad_input

    return qfn().apply


class phase_quantize_fn_cuda(nn.Module):
    def __init__(self, p_bit):
        super(phase_quantize_fn_cuda, self).__init__()
        assert p_bit <= 8 or p_bit == 32
        self.p_bit = p_bit
        self.uniform_q = uniform_quantize(k=p_bit)
        self.pi = 3.141592653589

    def forward(self, x):
        if self.p_bit == 32:
            phase_q = x
        elif self.w_bit == 1:
            E = torch.mean(torch.abs(x)).detach()
            phase_q = self.uniform_q(x / E) * E
        else:
            # phase = torch.tanh(x)
            # phase = phase / 2 / torch.max(torch.abs(phase)) + 0.5
            phase = phase / 2 / self.pi + 0.5
            # phase_q = 2 * self.uniform_q(phase) - 1
            phase_q = self.uniform_q(phase) * 2 * self.pi - self.pi
        return phase_q


class weight_quantize_fn(nn.Module):
    def __init__(self, w_bit, mode="oconv", alg="dorefa"):
        super(weight_quantize_fn, self).__init__()
        assert w_bit <= 32  # or w_bit == 32
        self.w_bit = w_bit
        self.alg = alg
        self.mode = mode
        assert alg in {
            "dorefa", "qnn"}, "[E] Only support Dorefa and QNN Algorithms"
        self.uniform_q = uniform_quantize(k=w_bit, gradient_clip=True)

    def forward(self, x):
        if self.w_bit == 32:
            # weight_q = x
            weight_q = torch.tanh(x)
            weight_q = weight_q / torch.max(torch.abs(weight_q))
            # weight = torch.tanh(x)
            # weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
            # weight_q = 2 * self.uniform_q(weight) - 1
        elif self.w_bit == 1:
            # weight = torch.tanh(x)
            # weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
            # weight_q = self.uniform_q(weight) * 0.5
            if(self.mode == "ringonn"):
                weight_q = (self.uniform_q(x) / 4) + 0.5
            else:
                E = x.data.abs().mean()
                weight_q = (self.uniform_q(x / E) * E + E) / 2
        else:
            if(self.alg == "dorefa"):
                weight = torch.tanh(x)
                weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
                weight_q = self.uniform_q(weight)
            elif(self.alg == "qnn"):
                x_min = torch.min(x).detach()
                x_max = torch.max(x).detach()
                x_range = x_max - x_min
                x = (x - x_min) / x_range
                x_q = self.uniform_q(x)
                weight_q = x_q * x_range + x_min
            else:
                assert NotImplementedError

        return weight_q


class activation_quantize_fn(nn.Module):
    def __init__(self, a_bit):
        super(activation_quantize_fn, self).__init__()
        assert a_bit <= 8 or a_bit == 32
        self.a_bit = a_bit
        self.uniform_q = uniform_quantize(k=a_bit)

    def forward(self, x):
        if self.a_bit == 32:
            activation_q = x
        else:
            activation_q = self.uniform_q(torch.clamp(x, 0, 1))
            # print(np.unique(activation_q.detach().numpy()))
        return activation_q


class input_quantize_fn(nn.Module):
    def __init__(self, in_bit, input_channel, mode="oconv", device=torch.device("cuda:0")):
        super(input_quantize_fn, self).__init__()
        assert in_bit <= 32
        self.in_bit = in_bit
        self.uniform_q = uniform_quantize(k=in_bit)
        self.device = device
        self.input_channel = input_channel
        self.mode = mode
        # self.E = Parameter(torch.zeros(input_channel,dtype=torch.float32, device=self.device) + 0.5)
        # self.E = Parameter(torch.zeros(1,dtype=torch.float32, device=self.device) + 0.5)
        self.gamma = 0.999

    def forward(self, x):
        if self.in_bit == 32:
            input_q = x
        elif(self.in_bit == 1):
            x = x.clamp(0,1)
            # if(x.dim() == 4):
            #     x_mean = x.data.mean(dim=(0,2,3))
            #     E = self.E.data.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            # elif(x.dim() == 2):
            #     x_mean = x.data.mean(dim=0)
            #     E = self.E.data.unsqueeze(0)
            if(self.mode == "oadder"):
                input_q = (self.uniform_q(x - 0.5) + 1) / 4
            if(self.mode == "ringonn"):
                input_q = (self.uniform_q(x - 0.5) + 1) / 2
            else:
                input_q = (self.uniform_q(x - 0.5) + 1) / 2
            # self.E.data.copy_(self.gamma * self.E.data + (1-self.gamma) * x_mean)

        else:
            input_q = self.uniform_q(torch.clamp(x, 0, 1))
            # print(np.unique(input_q.detach().numpy()))
        return input_q


def conv2d_Q_fn(w_bit):
    class Conv2d_Q(nn.Conv2d):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                     padding=0, dilation=1, groups=1, bias=True):
            super(Conv2d_Q, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, input, order=None):
            weight_q = self.quantize_fn(self.weight)
            # print(np.unique(weight_q.detach().numpy()))
            return F.conv2d(input, weight_q, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)

    return Conv2d_Q


def linear_Q_fn(w_bit):
    class Linear_Q(nn.Linear):
        def __init__(self, in_features, out_features, bias=True):
            super(Linear_Q, self).__init__(in_features, out_features, bias)
            self.w_bit = w_bit
            self.quantize_fn = weight_quantize_fn(w_bit=w_bit)

        def forward(self, input):
            weight_q = self.quantize_fn(self.weight)
            # print(np.unique(weight_q.detach().cpu().numpy()))
            return F.linear(input, weight_q, self.bias)

    return Linear_Q


class adder2d(nn.Module):

    def __init__(self, input_channel, output_channel, kernel_size, stride=1, padding=0, bias=False):
        super(adder2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.adder = torch.nn.Parameter(nn.init.normal_(torch.randn(
            output_channel, input_channel, kernel_size, kernel_size)))
        self.bias = bias
        if bias:
            self.b = torch.nn.Parameter(
                nn.init.uniform_(torch.zeros(output_channel)))


    def forward(self, x):
        output = adder2d_function(x, self.adder, self.stride, self.padding)
        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        return output


class OLinear_Q(nn.Module):
    def __init__(
        self,
        input_channel,
        output_channel,
        in_bit=16,
        w_bit=16,
        bias=False,
        mode="oconv",
        input_augment=False,
        device=torch.device("cuda:0")
    ):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.device = device
        self.mode = mode
        self.input_quantizer = input_quantize_fn(in_bit, self.input_channel, self.mode, self.device)
        self.weight_quantizer = weight_quantize_fn(w_bit, self.mode)
        self.weight = torch.nn.Parameter(torch.zeros(output_channel, input_channel, device=self.device))
        # self.mode = "oconv"
        self.phases = torch.cat([torch.zeros(1, 1, input_channel//2, device=self.device) + (np.pi / 2 if self.mode == "oconv_posw" else -np.pi/2), torch.zeros(1, 1, input_channel-input_channel//2, device=self.device) + np.pi / 2],dim=2)
        self.disks = torch.ones(1, 1, input_channel, 2, device=self.device)

        self.bias = bias
        self.input_augment = input_augment
        if(input_augment):
            self.beta = Parameter(torch.ones(1, self.input_channel).to(self.device))

        if bias:
            self.b = torch.nn.Parameter(torch.zeros(output_channel, device=self.device))
        else:
            self.b = None
        self.reset_parameters()
        self.calibration = False
        self.assigned = False
        self.func = {"oconv":self.optical_linear,
                     "oadder":self.optical_linear,
                     "oconv_posw":self.optical_linear,
                     "conv":self.linear,
                     "ringonn": self.ring_linear}[self.mode]
        if(self.mode == "ringonn"):
            self.ring_noise = None
            self.ring_crosstalk_perc = 0

    def reset_parameters(self):
        set_torch_deterministic()
        if(self.mode == "ringonn"):
            # nn.init.uniform_(self.weight, 1e-4, 1)
            # self.weight.data = self.weight.data.sqrt() * 2 - 1
            nn.init.kaiming_normal_(self.weight)
        else:
            nn.init.kaiming_normal_(self.weight)
        # _, self.scale = quant_kaiming_uniform_(self.weight, self.w_bit, beta=1.5)

        if self.bias:
            nn.init.uniform_(self.b)

    def assign_engines(self, out_par=1, batch_par=1, phase_noise_std=0, disk_noise_std=0, deterministic=False):
        out_par = math.gcd(out_par, self.output_channel)
        if(phase_noise_std > 1e-4):
            if(deterministic):
                set_torch_deterministic()
            mean_1 = np.pi/2 if self.mode == "oconv_posw" else -np.pi/2
            self.phases = torch.cat([torch.zeros(out_par, batch_par, self.input_channel//2, device=self.device).normal_(mean=mean_1, std=phase_noise_std).clamp(mean_1-3*phase_noise_std, mean_1+3*phase_noise_std), torch.zeros(out_par, batch_par, self.input_channel-self.input_channel//2, device=self.device).normal_(mean=np.pi/2, std=phase_noise_std).clamp(np.pi/2-3*phase_noise_std, np.pi/2+3*phase_noise_std)], dim=2)
        else:
            self.phases = torch.cat([torch.zeros(1, 1, self.input_channel//2, device=self.device) + (np.pi / 2 if self.mode == "oconv_posw" else -np.pi/2), torch.zeros(1, 1, self.input_channel-self.input_channel//2, device=self.device) + np.pi / 2],dim=2)
        if(disk_noise_std > 1e-5):
            if(deterministic):
                set_torch_deterministic()
            self.disks = 1 - torch.zeros(out_par, batch_par, self.input_channel, 2, device=self.device).normal_(mean=0, std=disk_noise_std).clamp(-3*disk_noise_std, 3*disk_noise_std).abs()
        else:
            self.disks = torch.ones(1, 1, self.input_channel, 2, device=self.device)
        self.assigned = True

    def deassign_engines(self):
        self.phases = torch.cat([torch.zeros(1, 1, self.input_channel//2, device=self.device) + (np.pi / 2 if self.mode == "oconv_posw" else -np.pi/2), torch.zeros(1, 1, self.input_channel-self.input_channel//2, device=self.device) + np.pi / 2],dim=2)
        self.disks = torch.ones(1, 1, self.input_channel, 2, device=self.device)
        self.assigned = False

    def robust_reassign_engines(self):
        W = self.weight.data
        n_out_tile, out_tile_size = self.phases.size(0), W.size(0) // self.phases.size(0)
        n_image_tile = self.phases.size(1)
        W_norm = W.view(n_out_tile, -1).norm(p=2, dim=-1) # [n_tile]
        _, W_indices = W_norm.sort()
        # self.beta = (self.phases.sin().abs()*(self.disks[...,0]+self.disks[...,1])/2).mean(dim=2) # [out_par, image_par]
        self.beta = (self.disks[...,0]-self.disks[...,1]).abs().mean(dim=2)
        _, engine_indices = self.beta.view(-1).sort(descending=True)
        self.phases = self.phases.view(n_out_tile*n_image_tile, self.phases.size(2))[engine_indices, ...].view_as(self.phases)[W_indices, ...]
        self.disks = self.disks.view(n_out_tile*n_image_tile, self.disks.size(2), self.disks.size(3))[engine_indices, ...].view_as(self.disks)[W_indices, ...]

    def robust_reassign_engines_fine(self):
        for start, end in [(0, self.input_channel//2), (self.input_channel//2, self.input_channel)]:
            W = self.weight.data[:,start:end,...]
            n_out_tile, out_tile_size = self.phases.size(0), W.size(0) // self.phases.size(0)
            n_image_tile = self.phases.size(1)
            W_norm = W.view(n_out_tile, out_tile_size, -1).norm(p=2, dim=1) # [n_tile, inc*ks*ks]
            _, W_indices = W_norm.sort(dim=-1)
            # W_indices = W_indices.unsqueeze(1).expand(1,n_image_tile,1)

            self.beta = (self.phases[:,:,start:end,...].sin().abs()*(self.disks[:,:,start:end,...,0]+self.disks[:,:,start:end,...,1])) # [out_par, image_par, inc]
            # self.beta = (self.disks[:,:,start:end,...,0]-self.disks[:,:,start:end,...,1]).abs() # [out_par, image_par, inc]

            _, engine_indices = self.beta.view(n_out_tile, n_image_tile, -1).sort(descending=True, dim=-1)
            phases = self.phases[:,:,start:end,...].view(n_out_tile, n_image_tile, -1)
            self.phases[:,:,start:end,...] = phases.scatter(-1, W_indices.unsqueeze(1).expand(-1,n_image_tile,-1), phases.gather(-1, engine_indices)).view_as(self.phases[:,:,start:end,...])
            engine_indices = engine_indices.unsqueeze(-1).expand(-1,-1,-1,2)
            W_indices = W_indices.unsqueeze(1).unsqueeze(-1).expand(-1,n_image_tile,-1,2)
            # print(self.disks)
            disks = self.disks[:,:,start:end,...].view(n_out_tile, n_image_tile, -1, 2)
            self.disks[:,:,start:end,...] = disks.scatter(-2, W_indices, disks.gather(-2, engine_indices)).view_as(self.disks[:,:,start:end,...])

    def assign_ring_noise(self, ring_noise_std=0, ring_crosstalk_perc=0, deterministic=False):
        if(deterministic):
            set_torch_deterministic()
        self.ring_noise = torch.zeros(self.output_channel, self.input_channel, device=self.device, dtype=torch.float32).normal_(0, std=ring_noise_std).clamp_(-3*ring_noise_std,3*ring_noise_std)
        self.ring_crosstalk_perc = ring_crosstalk_perc

    def static_pre_calibration(self):
        coeff = 0.25
        N = self.phases.size(2)
        sin_phi = self.phases.sin()
        self.beta0 = self.disks[...,0].sum(dim=(2)) # [out_par, image_par]
        self.beta1 = self.disks[...,1].sum(dim=(2)) # [out_par, image_par]
        self.beta2 = (sin_phi.abs()*(self.disks[...,0]+self.disks[...,1])/2).mean(dim=2) # [out_par, image_par]
        self.correction_disk_0 = self.disks[...,0].mean(dim=2)
        self.correction_disk_1 = self.disks[...,1].mean(dim=2)
        self.correction_phi = self.beta2 / self.disks.mean(dim=(2,3))

        # return self.correction # [out_par, image_par]

    def enable_calibration(self):
        self.calibration = True

    def disable_calibration(self):
        self.calibration = False

    def apply_calibration(self, x):
        # x [outc, bs]
        x_size = x.size()
        bs = x.size(0)
        n_out_tile, out_tile_size = self.phases.size(0), x.size(0) // self.phases.size(0)
        n_batch_tile, batch_tile_size = self.phases.size(1), x.size(1) // self.phases.size(1)
        correction = self.correction.unsqueeze(1).unsqueeze(-1) # [out_par, 1, image_par, 1]
        x = x.view(n_out_tile, out_tile_size, n_batch_tile, batch_tile_size)
        x = x / correction
        x = x.view(x_size)
        return x

    def tiling_matmul(self, W, X, sin_phases, disks):
        n_out_tile, out_tile_size = sin_phases.size(0), W.size(0) // sin_phases.size(0)
        n_batch_tile, batch_tile_size = sin_phases.size(1), X.size(1) // sin_phases.size(1)
        W = W.unsqueeze(1) # [outc, 1, in_c]
        W = W.view(n_out_tile, out_tile_size, 1, -1) # [n_out_tile,out_tile_size,1, in_c]
        X = X.view(X.size(0), n_batch_tile, batch_tile_size).permute(1, 2, 0).contiguous().unsqueeze(0) # [1, n_batch_tile, batch_tile_size, inc]

        factor_mul = sin_phases * (disks[...,0]+disks[...,1]) / 2
        factor_mul = factor_mul.view(n_out_tile, n_batch_tile, -1).unsqueeze(1) # [n_out_tile,1,n_batch_tile, in_c]

        # inject additive error
        factor_add = (disks[..., 0] - disks[..., 1]) / 4
        factor_add = factor_add.view(n_out_tile, n_batch_tile, -1).unsqueeze(1) # [n_out_tile,1,n_batch_tile, in_c]
        additive_error = ((W * W) * factor_add).sum(dim=-1).unsqueeze(3) # [n_out_tile, out_tile_size, n_batch_tile, 1]
        factor_add = factor_add.squeeze(1).unsqueeze(2) # [n_out_tile, n_batch_tile, 1, in_c]
        additive_error = additive_error + ((X * X) * factor_add).sum(dim=-1).unsqueeze(1) # [n_out_tile, out_tile_size, n_batch_tile, 1]+[n_out_tile, 1, n_batch_tile, batch_tile_size]->[n_out_tile, out_tile_size, n_batch_tile, batch_tile_size]
        additive_error = additive_error.view(self.output_channel, -1) # [outc, bs]

        W = (W * factor_mul).view(self.output_channel, n_batch_tile, -1).permute(1,0,2).contiguous() # # [n_batch_tile,outc,in_c]
        X = X.squeeze(0).permute(0,2,1).contiguous() # X [n_batch_tile, in_c, batch_tile_size]

        out = W.bmm(X).transpose(0,1).contiguous().view(self.output_channel, -1) # [outc, bs]
        out = out + additive_error

        return out

    def tiling_matmul_with_calibration(self, W, X, sin_phases, disks):
        n_out_tile, out_tile_size = sin_phases.size(0), W.size(0) // sin_phases.size(0)
        n_batch_tile, batch_tile_size = sin_phases.size(1), X.size(1) // sin_phases.size(1)
        disks = self.disks.view(n_out_tile, n_batch_tile, -1 , 2).unsqueeze(1) # [n_out_tile, 1, n_batch_tile, inc, 2]
        W = W.unsqueeze(1) # [outc, 1, in_c]
        W = W.view(n_out_tile, out_tile_size, 1, -1) # [n_out_tile,out_tile_size,1, in_c]
        # W_sq = (W * W).unsqueeze(-1) # [n_out_tile,out_tile_size,1, in_c, 1]
        W_sq_sum = ((W * W).unsqueeze(-1)*disks).sum(dim=-2).unsqueeze(3) # [n_out_tile, out_tile_size, n_batch_tile, 1, 2] for two rails

        X = X.view(X.size(0), n_batch_tile, batch_tile_size).permute(1, 2, 0).contiguous().unsqueeze(0) # [1, n_batch_tile, batch_tile_size, inc]
        # X_sq = (X * X).unsqueeze(-1)
        # disks = disks.transpose(1,2).contiguous() # [n_out_tile, n_batch_tile, 1, inc, 2]
        X_sq_sum = ((X * X).unsqueeze(-1) * disks.transpose(1,2).contiguous()).sum(dim=-2).unsqueeze(1) # [n_out_tile,1, n_batch_tile, batch_tile_size, 2]

        factor_mul = sin_phases.unsqueeze(-1) * self.disks
        factor_mul = factor_mul.view(n_out_tile, n_batch_tile, -1, 2).unsqueeze(1) # [n_out_tile,1,n_batch_tile, in_c, 2]

        W = (W.unsqueeze(-1) * factor_mul).view(self.output_channel, n_batch_tile, -1, 2).permute(1,0,2,3).contiguous() # # [n_batch_tile,outc,in_c, 2]
        X = X.squeeze(0).permute(0,2,1).contiguous() # X [n_batch_tile, in_c, batch_tile_size]
        two_X_W_sin_phi_alpha_0 = 2*W[...,0].bmm(X).view(n_batch_tile, n_out_tile, out_tile_size, batch_tile_size).permute(1,2,0,3).contiguous()
        two_X_W_sin_phi_alpha_1 = 2*W[...,1].bmm(X).view(n_batch_tile, n_out_tile, out_tile_size, batch_tile_size).permute(1,2,0,3).contiguous() #  [n_out_tile, out_tile_size, n_batch_tile, batch_tile_size]
        W_sq_plus_X_sq_sum = W_sq_sum + X_sq_sum # [n_out_tile, out_tile_size, n_batch_tile, batch_tile_size, 2]
        rail_0 = (W_sq_plus_X_sq_sum[..., 0] + two_X_W_sin_phi_alpha_0) / (4 * self.correction_disk_0.unsqueeze(1).unsqueeze(-1))
        rail_1 = (W_sq_plus_X_sq_sum[..., 1] - two_X_W_sin_phi_alpha_1) / (4 * self.correction_disk_1.unsqueeze(1).unsqueeze(-1))
        # rail_0 = (W_sq_plus_X_sq_sum[..., 0] + two_X_W_sin_phi_alpha_0) / 4
        # rail_1 = (W_sq_plus_X_sq_sum[..., 1] - two_X_W_sin_phi_alpha_1) / 4
        # print(rail_0.size(), rail_1.size())
        out = (rail_0 - rail_1) / self.correction_phi.unsqueeze(1).unsqueeze(-1)
        out = out.view(self.output_channel, -1)

        return out

    def optical_linear(self, X, W):
        if(self.assigned == False):
            ## half kernels are negative
            if(self.mode != "oconv_posw"):
                W = torch.cat([-W[..., :self.input_channel//2], W[..., self.input_channel//2:]], dim=1)
            return F.linear(X, W, bias=None)
        n_x = X.size(0)
        X_col = X.permute(1,0).contiguous() # [inc, bs]

        # matmul with phase consideration
        # out = W.matmul(X_col)
        if(self.calibration):
            out = self.tiling_matmul_with_calibration(W, X_col, self.phases.sin(), self.disks)
        else:
            out = self.tiling_matmul(W, X_col, self.phases.sin(), self.disks)
        # print(1, F.mse_loss(real, out).data.item())
        # print("real:", real[0,100])
        # print("noisy:", out[0,100])
        # [outc, h*w*bs]

        # print("calib:", out[0,100])
        # print(2, F.mse_loss(real, out).data.item())
        out = out.permute(1, 0).contiguous()
        return out

    def linear(self, X, W):
        W = W * 2 - 1
        return F.linear(X, W, bias=self.b)

    def ring_linear(self, X, W):
        if(self.ring_noise is not None):
            W += self.ring_noise
        W = W.clamp(0, 1)
        if(self.ring_crosstalk_perc > 1e-6):
            kernel = torch.from_numpy(np.array([[0, self.ring_crosstalk_perc, 0],[self.ring_crosstalk_perc,0,self.ring_crosstalk_perc],[0, self.ring_crosstalk_perc, 0]])).float().to(self.device)
            W += F.conv2d(W.unsqueeze(0).unsqueeze(0).abs(), kernel.unsqueeze(0).unsqueeze(0), bias=None, padding=1, stride=1).squeeze()
        W = W.clamp(0, 1)
        W = 2*W - 1
        # print(W, X*X)

        return F.linear(X*X, W, bias=self.b)

    def forward(self, x):
        input = x.clamp(0,1)
        x = self.input_quantizer(x)
        if(self.input_augment and self.beta.data.max() > 1e-6):
            x = (1-self.beta) * x + self.beta * input
        weight = self.weight_quantizer(self.weight)

        output = self.func(x, weight)

        if self.bias:
            output += self.b.unsqueeze(0)
        # output = output / weight.norm(p=2, dim=1) / 1.21
        return output


class OAdder2d_Q(nn.Module):
    def __init__(
        self,
        input_channel,
        output_channel,
        kernel_size,
        stride=1,
        padding=0,
        in_bit=16,
        w_bit=16,
        mode="oconv",
        bias=False,
        input_augment=False,
        device=torch.device("cuda:0")
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.in_bit = in_bit
        self.w_bit = w_bit
        self.mode = mode
        self.device = device
        self.input_quantizer = input_quantize_fn(in_bit, self.input_channel, self.mode, self.device)
        self.weight_quantizer = weight_quantize_fn(w_bit, self.mode)
        self.weight = torch.nn.Parameter(torch.zeros(output_channel, input_channel, kernel_size, kernel_size, device=self.device))

        self.phases = torch.cat([torch.zeros(1, 1, input_channel//2, kernel_size, kernel_size, device=self.device) + (np.pi / 2 if self.mode in {"oadder", "oconv_posw"} else -np.pi/2), torch.zeros(1, 1, input_channel-input_channel//2, kernel_size, kernel_size, device=self.device) + np.pi / 2],dim=2)
        self.disks = torch.ones(1, 1, input_channel, kernel_size, kernel_size, 2, device=self.device)
        # if(self.mode in {"oadder", "oconv_posw"}):
        #     self.phases.abs_()

        self.bias = bias
        self.input_augment = input_augment
        if(input_augment):
            self.beta = Parameter(torch.ones(1, self.input_channel, 1, 1).to(self.device))


        if bias:
            self.b = torch.nn.Parameter(torch.zeros(output_channel, device=self.device))
        else:
            self.b = None
        self.adder_func = {"adder": self.adder2d_function,
                           "oadder": self.optical_adder2d_function,
                           "oconv": self.optical_conv2d_function,
                           "oconv_posw": self.optical_conv2d_function,
                           "conv": self.conv2d_function,
                           "ringonn": self.ring_conv2d_function}[mode]
        self.reset_parameters()
        self.calibration = False
        self.assigned = False
        if(self.mode == "ringonn"):
            self.ring_noise = None
            self.ring_crosstalk_perc = 0
        # self.alpha = Parameter(torch.ones(1, self.output_channel, 1, 1).to(self.device))

    def reset_parameters(self):
        # set_torch_deterministic()
        if(self.mode == "ringonn"):
            # nn.init.uniform_(self.weight, -1, 2)
            nn.init.kaiming_normal_(self.weight)
        else:
            nn.init.kaiming_normal_(self.weight)
        # _, self.scale = quant_kaiming_uniform_(self.weight, self.w_bit, beta=1.5)
        if self.bias:
            nn.init.uniform_(self.b)


    def assign_engines(self, out_par=1, image_par=1, phase_noise_std=0, disk_noise_std=0, deterministic=False):
        out_par = math.gcd(out_par, self.output_channel)
        if(phase_noise_std > 1e-4):
            if(deterministic):
                set_torch_deterministic()
            mean_1 = np.pi/2 if self.mode in {"oadder", "oconv_posw"} else -np.pi/2
            self.phases = torch.cat([torch.zeros(out_par, image_par, self.input_channel//2, self.kernel_size, self.kernel_size, device=self.device).normal_(mean=mean_1, std=phase_noise_std).clamp(mean_1-3*phase_noise_std, mean_1+3*phase_noise_std), torch.zeros(out_par, image_par, self.input_channel-self.input_channel//2, self.kernel_size, self.kernel_size, device=self.device).normal_(mean=np.pi/2, std=phase_noise_std).clamp(np.pi/2-3*phase_noise_std, np.pi/2+3*phase_noise_std)], dim=2)
        else:
            self.phases = torch.cat([torch.zeros(1, 1, self.input_channel//2, self.kernel_size, self.kernel_size, device=self.device) + (np.pi/2 if self.mode in {"oadder", "oconv_posw"} else -np.pi/2), torch.zeros(1, 1, self.input_channel-self.input_channel//2, self.kernel_size, self.kernel_size, device=self.device) + np.pi / 2],dim=2)


        if(disk_noise_std > 1e-5):
            if(deterministic):
                set_torch_deterministic()
            self.disks = 1 - torch.zeros(out_par, image_par, self.input_channel, self.kernel_size, self.kernel_size, 2, device=self.device).normal_(mean=0, std=disk_noise_std).clamp(-3*disk_noise_std, 3*disk_noise_std).abs()
        else:
            self.disks = torch.ones(1, 1, self.input_channel, self.kernel_size, self.kernel_size, 2, device=self.device)
        self.assigned = True

    def deassign_engines(self):
        self.phases = torch.cat([torch.zeros(1, 1, self.input_channel//2, self.kernel_size, self.kernel_size, device=self.device) + (np.pi/2 if self.mode in {"oadder", "oconv_posw"} else -np.pi/2), torch.zeros(1, 1, self.input_channel-self.input_channel//2, self.kernel_size, self.kernel_size, device=self.device) + np.pi / 2],dim=2)
        self.disks = torch.ones(1, 1, self.input_channel, self.kernel_size, self.kernel_size, 2, device=self.device)
        self.assigned = False

    def robust_reassign_engines(self):
        W = self.weight.data
        n_out_tile, out_tile_size = self.phases.size(0), W.size(0) // self.phases.size(0)
        n_image_tile = self.phases.size(1)
        W_norm = W.view(n_out_tile, -1).norm(p=2, dim=-1) # [n_tile]
        _, W_indices = W_norm.sort()
        # print(W_norm, W_indices)
        # self.beta = (self.phases.sin().abs()*(self.disks[...,0]+self.disks[...,1])/2).mean(dim=(2,3,4)) # [out_par, image_par]
        self.beta = (self.disks[...,0]-self.disks[...,1]).abs().mean(dim=(2,3,4))
        _, engine_indices = self.beta.view(-1).sort(descending=True)
        # print(self.beta.abs().view(-1), engine_indices)
        # print("before", self.phases)

        self.phases = self.phases.view(n_out_tile*n_image_tile, self.phases.size(2), self.phases.size(3), self.phases.size(4))[engine_indices, ...].view_as(self.phases)[W_indices, ...]
        # print("after", self.phases)
        self.disks = self.disks.view(n_out_tile*n_image_tile, self.disks.size(2), self.disks.size(3), self.disks.size(4), self.disks.size(5))[engine_indices, ...].view_as(self.disks)[W_indices, ...]

    def robust_reassign_engines_fine(self):

        for start, end in [(0, self.input_channel//2), (self.input_channel//2, self.input_channel)]:
            W = self.weight.data[:,start:end,...]
            n_out_tile, out_tile_size = self.phases.size(0), W.size(0) // self.phases.size(0)
            n_image_tile = self.phases.size(1)
            W_norm = W.view(n_out_tile, out_tile_size, -1).norm(p=2, dim=1) # [n_tile, inc*ks*ks]
            _, W_indices = W_norm.sort(dim=-1)
            # W_indices = W_indices.unsqueeze(1).expand(1,n_image_tile,1)
            self.beta = (self.phases[:,:,start:end,...].sin().abs()*(self.disks[:,:,start:end,...,0]+self.disks[:,:,start:end,...,1])) # [out_par, image_par, inc*ks*ks]
            # self.beta = (self.disks[:,:,start:end,...,0]-self.disks[:,:,start:end,...,1]).abs() # [out_par, image_par, inc]

            _, engine_indices = self.beta.view(n_out_tile, n_image_tile, -1).sort(descending=True, dim=-1)
            # self.phases[:,:,start:end,...] = self.phases[:,:,start:end,...].view(n_out_tile, n_image_tile, -1).gather(-1, engine_indices).gather(-1, W_indices.unsqueeze(1).expand(-1,n_image_tile,-1)).view_as(self.phases[:,:,start:end,...])
            phases = self.phases[:,:,start:end,...].view(n_out_tile, n_image_tile, -1)
            self.phases[:,:,start:end,...] = phases.scatter(-1, W_indices.unsqueeze(1).expand(-1,n_image_tile,-1), phases.gather(-1, engine_indices)).view_as(self.phases[:,:,start:end,...])
            engine_indices = engine_indices.unsqueeze(-1).expand(-1,-1,-1,2)
            W_indices = W_indices.unsqueeze(1).unsqueeze(-1).expand(-1,n_image_tile,-1,2)
            # print(self.disks)
            disks = self.disks[:,:,start:end,...].view(n_out_tile, n_image_tile, -1, 2)
            self.disks[:,:,start:end,...] = disks.scatter(-2, W_indices, disks.gather(-2, engine_indices)).view_as(self.disks[:,:,start:end,...])
            # print(self.disks)

    def static_pre_calibration(self):
        coeff = 0.25
        N = self.phases.size(2)*self.phases.size(3)*self.phases.size(4)
        sin_phi = self.phases.sin()
        self.beta0 = self.disks[...,0].sum(dim=(2,3,4)) # [out_par, image_par]
        self.beta1 = self.disks[...,1].sum(dim=(2,3,4)) # [out_par, image_par]
        self.beta2 = (sin_phi.abs()*(self.disks[...,0]+self.disks[...,1])/2).mean(dim=(2,3,4)) # [out_par, image_par]
        if(self.mode == "oadder"):
            self.correction = 2 * N * coeff * (1 - self.beta2)
        elif(self.mode == "oconv"):
            self.correction_disk_0 = self.disks[...,0].mean(dim=(2,3,4))
            self.correction_disk_1 = self.disks[...,1].mean(dim=(2,3,4))
            self.correction_phi = self.beta2 / self.disks.mean(dim=(2,3,4,5))
        else:
            raise NotImplementedError
        # return self.correction # [out_par, image_par]

    def enable_calibration(self):
        self.calibration = True

    def disable_calibration(self):
        self.calibration = False

    def apply_calibration(self, x):
        # x [outc, h*w*bs]

        x_size = x.size()
        bs = x.size(0)
        n_out_tile, out_tile_size = self.phases.size(0), x.size(0) // self.phases.size(0)
        n_image_tile, image_tile_size = self.phases.size(1), x.size(1) // self.phases.size(1)
        # correction = self.correction.unsqueeze(1).unsqueeze(-1) # [out_par, 1, image_par, 1]
        x = x.view(n_out_tile, out_tile_size, n_image_tile, image_tile_size)
        if(self.mode == "oconv"):
            correction_disk_0 = self.correction_disk_0.unsqueeze(1).unsqueeze(-1) # [out_par, 1, image_par, 1]
            correction_disk_1 = self.correction_disk_1.unsqueeze(1).unsqueeze(-1) # [out_par, 1, image_par, 1]
            correction_phi = self.correction_phi.unsqueeze(1).unsqueeze(-1) # [out_par, 1, image_par, 1]

        elif(self.mode == "oadder"):
            x = x + correction
        else:
            raise NotImplementedError
        x = x.view(x_size)
        return x

    def assign_ring_noise(self, ring_noise_std=0, ring_crosstalk_perc=0, deterministic=False):
        if(deterministic):
            set_torch_deterministic()
        self.ring_noise = torch.zeros(self.output_channel, self.input_channel*self.kernel_size*self.kernel_size, device=self.device, dtype=torch.float32).normal_(0, std=ring_noise_std).clamp_(-3*ring_noise_std,3*ring_noise_std)
        self.ring_crosstalk_perc = ring_crosstalk_perc

    def tiling_L2_distance(self, W, X, sin_phases, disks):
        n_out_tile, out_tile_size = sin_phases.size(0), W.size(0) // sin_phases.size(0)
        n_image_tile, image_tile_size = sin_phases.size(1), X.size(1) // sin_phases.size(1)
        W = W.unsqueeze(1) # [outc, 1, in_c*kernel_size*kernel_size]
        W = W.view(n_out_tile, out_tile_size, 1, -1) # [n_out_tile,out_tile_size,1, in_c*kernel_size*kernel_size]
        X = X.view(X.size(0), n_image_tile, image_tile_size).permute(1, 2, 0).contiguous().unsqueeze(0) # [1, n_image_tile, image_tile_size, inc*kernel_size*kernel_size]

        factor_mul = -2 * sin_phases * disks[...,1]
        factor_mul = factor_mul.view(n_out_tile, n_image_tile, -1).unsqueeze(1) # [n_out_tile,1,n_image_tile, in_c*kernel_size*kernel_size]

        # additive term
        factor_add = disks[..., 1]
        factor_add = factor_add.view(n_out_tile, n_image_tile, -1).unsqueeze(1) # [n_out_tile,1,n_image_tile, in_c*kernel_size*kernel_size]
        additive_error = ((W * W) * factor_add).sum(dim=-1).unsqueeze(3) # [n_out_tile, out_tile_size, n_image_tile, 1]
        factor_add = factor_add.squeeze(1).unsqueeze(2) # [n_out_tile, n_image_tile, 1, in_c*kernel_size*kernel_size]
        additive_error = additive_error + ((X * X) * factor_add).sum(dim=-1).unsqueeze(1) # [n_out_tile, out_tile_size, n_image_tile, 1]+[n_out_tile, 1, n_image_tile, image_tile_size]->[n_out_tile, out_tile_size, n_image_tile, image_tile_size]
        additive_error = additive_error.view(self.output_channel, -1) # [outc, h*w*bs]

        W = (W * factor_mul).view(self.output_channel, n_image_tile, -1).permute(1,0,2).contiguous() # # [n_image_tile,outc,in_c*kernel_size*kernel_size]
        X = X.squeeze(0).permute(0,2,1).contiguous() # X [n_image_tile, in_c*kernel_size*kernel_size, image_tile_size]

        out = W.bmm(X).transpose(0,1).contiguous().view(self.output_channel, -1) # [outc, h*w*bs]
        out = out + additive_error
        return out

    def tiling_matmul(self, W, X, sin_phases, disks):
        n_out_tile, out_tile_size = sin_phases.size(0), W.size(0) // sin_phases.size(0)
        n_image_tile, image_tile_size = sin_phases.size(1), X.size(1) // sin_phases.size(1)
        W = W.unsqueeze(1) # [outc, 1, in_c*kernel_size*kernel_size]
        W = W.view(n_out_tile, out_tile_size, 1, -1) # [n_out_tile,out_tile_size,1, in_c*kernel_size*kernel_size]
        X = X.view(X.size(0), n_image_tile, image_tile_size).permute(1, 2, 0).contiguous().unsqueeze(0) # [1, n_image_tile, image_tile_size, inc*kernel_size*kernel_size]

        factor_mul = sin_phases * (disks[...,0]+disks[...,1]) / 2
        factor_mul = factor_mul.view(n_out_tile, n_image_tile, -1).unsqueeze(1) # [n_out_tile,1,n_image_tile, in_c*kernel_size*kernel_size]

        # inject additive error
        factor_add = (disks[..., 0] - disks[..., 1]) / 4
        factor_add = factor_add.view(n_out_tile, n_image_tile, -1).unsqueeze(1) # [n_out_tile,1,n_image_tile, in_c*kernel_size*kernel_size]
        additive_error = ((W * W) * factor_add).sum(dim=-1).unsqueeze(3) # [n_out_tile, out_tile_size, n_image_tile, 1]
        factor_add = factor_add.squeeze(1).unsqueeze(2) # [n_out_tile, n_image_tile, 1, in_c*kernel_size*kernel_size]
        additive_error = additive_error + ((X * X) * factor_add).sum(dim=-1).unsqueeze(1) # [n_out_tile, out_tile_size, n_image_tile, 1]+[n_out_tile, 1, n_image_tile, image_tile_size]->[n_out_tile, out_tile_size, n_image_tile, image_tile_size]
        additive_error = additive_error.view(self.output_channel, -1) # [outc, h*w*bs]

        W = (W * factor_mul).view(self.output_channel, n_image_tile, -1).permute(1,0,2).contiguous() # # [n_image_tile,outc,in_c*kernel_size*kernel_size]
        X = X.squeeze(0).permute(0,2,1).contiguous() # X [n_image_tile, in_c*kernel_size*kernel_size, image_tile_size]

        out = W.bmm(X).transpose(0,1).contiguous().view(self.output_channel, -1) # [outc, h*w*bs]
        out = out + additive_error
        return out

    def tiling_matmul_with_oconv_calibration(self, W, X, sin_phases, disks):
        n_out_tile, out_tile_size = sin_phases.size(0), W.size(0) // sin_phases.size(0)
        n_image_tile, image_tile_size = sin_phases.size(1), X.size(1) // sin_phases.size(1)
        disks = self.disks.view(n_out_tile, n_image_tile, -1 , 2).unsqueeze(1) # [n_out_tile, 1, n_image_tile, inc*ks*ks, 2]
        W = W.unsqueeze(1) # [outc, 1, in_c*kernel_size*kernel_size]
        W = W.view(n_out_tile, out_tile_size, 1, -1) # [n_out_tile,out_tile_size,1, in_c*kernel_size*kernel_size]
        # W_sq = (W * W).unsqueeze(-1) # [n_out_tile,out_tile_size,1, in_c*kernel_size*kernel_size, 1]
        W_sq_sum = ((W * W).unsqueeze(-1)*disks).sum(dim=-2).unsqueeze(3) # [n_out_tile, out_tile_size, n_image_tile, 1, 2] for two rails


        X = X.view(X.size(0), n_image_tile, image_tile_size).permute(1, 2, 0).contiguous().unsqueeze(0) # [1, n_image_tile, image_tile_size, inc*kernel_size*kernel_size]
        # X_sq = (X * X).unsqueeze(-1)
        # disks = disks.transpose(1,2).contiguous() # [n_out_tile, n_image_tile, 1, inc*ks*ks, 2]
        X_sq_sum = ((X * X).unsqueeze(-1) * disks.transpose(1,2).contiguous()).sum(dim=-2).unsqueeze(1) # [n_out_tile,1, n_image_tile, image_tile_size, 2]

        factor_mul = sin_phases.unsqueeze(-1) * self.disks
        factor_mul = factor_mul.view(n_out_tile, n_image_tile, -1, 2).unsqueeze(1) # [n_out_tile,1,n_image_tile, in_c*kernel_size*kernel_size, 2]

        W = (W.unsqueeze(-1) * factor_mul).view(self.output_channel, n_image_tile, -1, 2).permute(1,0,2,3).contiguous() # # [n_image_tile,outc,in_c*kernel_size*kernel_size, 2]
        X = X.squeeze(0).permute(0,2,1).contiguous() # X [n_image_tile, in_c*kernel_size*kernel_size, image_tile_size]
        two_X_W_sin_phi_alpha_0 = 2*W[...,0].bmm(X).view(n_image_tile, n_out_tile, out_tile_size, image_tile_size).permute(1,2,0,3).contiguous()
        two_X_W_sin_phi_alpha_1 = 2*W[...,1].bmm(X).view(n_image_tile, n_out_tile, out_tile_size, image_tile_size).permute(1,2,0,3).contiguous() #  [n_out_tile, out_tile_size, n_image_tile, image_tile_size]
        W_sq_plus_X_sq_sum = W_sq_sum + X_sq_sum # [n_out_tile, out_tile_size, n_image_tile, image_tile_size, 2]
        rail_0 = (W_sq_plus_X_sq_sum[..., 0] + two_X_W_sin_phi_alpha_0) / (4 * self.correction_disk_0.unsqueeze(1).unsqueeze(-1))
        rail_1 = (W_sq_plus_X_sq_sum[..., 1] - two_X_W_sin_phi_alpha_1) / (4 * self.correction_disk_1.unsqueeze(1).unsqueeze(-1))
        # rail_0 = (W_sq_plus_X_sq_sum[..., 0] + two_X_W_sin_phi_alpha_0) / 4
        # rail_1 = (W_sq_plus_X_sq_sum[..., 1] - two_X_W_sin_phi_alpha_1) / 4
        # print(rail_0.size(), rail_1.size())
        out = (rail_0 - rail_1) / self.correction_phi.unsqueeze(1).unsqueeze(-1)
        out = out.view(self.output_channel, -1)

        return out

    def adder2d_function(self, X, W, stride=1, padding=0):
        n_filters, d_filter, h_filter, w_filter = W.size()
        n_x, d_x, h_x, w_x = X.size()

        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(X.view(
            1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
        X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
        W_col = W.view(n_filters, -1)

        out = -torch.cdist(W_col, X_col.transpose(0, 1), 1)

        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()

        return out

    def optical_adder2d_function(self, X, W, stride=1, padding=0):
        if(self.assigned == False):
            ## half kernels are negative
            if(self.mode != "oconv_posw"):
                W = torch.cat([-W[:, 0:self.input_channel//2,...], W[:, self.input_channel//2:,...]], dim=1)
        n_filters, d_filter, h_filter, w_filter = W.size()
        n_x, d_x, h_x, w_x = X.size()

        h_out = (h_x - h_filter + 2 * padding) / stride + 1
        w_out = (w_x - w_filter + 2 * padding) / stride + 1

        h_out, w_out = int(h_out), int(w_out)
        X_col = torch.nn.functional.unfold(X.view(
            1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
        X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
        W_col = W.view(n_filters, -1) # [out_c, in_c*kernel_size*kernel_size]

        if(self.assigned == False):
            W_col_sq = (W_col * W_col).sum(dim=1, keepdim=True) # [outc, 1]
            X_col_sq = (X_col * X_col).sum(dim=0, keepdim=True) # [1, ...]
            W_by_X = -2*W_col.matmul(X_col) # [outc, inc*kernel_size*kernel_size]@ [inc*ks*ks, ...] = [outc, ...]
            out = -(W_col_sq + X_col_sq + W_by_X)
        else:
            out = -self.tiling_L2_distance(W_col, X_col, self.phases.sin(), self.disks)

        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()

        return out

    def optical_conv2d_function(self, X, W, stride=1, padding=0):
        if(self.assigned == False):
            ## half kernels are negative
            if(self.mode != "oconv_posw"):
                W = torch.cat([-W[:, 0:self.input_channel//2,...], W[:, self.input_channel//2:,...]], dim=1)
            return F.conv2d(X, W, stride=stride, padding=padding)
        n_x = X.size(0)
        n_filters, d_filter, h_filter, w_filter = W.size()
        W_col, X_col, h_out, w_out = im2col_2d(W, X, stride, padding)
        # matmul with phase consideration
        # real = W_col.matmul(X_col)
        if(self.calibration):
            out = self.tiling_matmul_with_oconv_calibration(W_col, X_col, self.phases.sin(), self.disks)
        else:
            out = self.tiling_matmul(W_col, X_col, self.phases.sin(), self.disks)
        # print(1, F.mse_loss(real, out).data.item())
        # print("real:", real[0,100])
        # print("noisy:", out[0,100])
        # [outc, h*w*bs]

        # print("calib:", out[0,100])
        # print(2, F.mse_loss(real, out).data.item())
        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()

        return out

    def conv2d_function(self, X, W, stride=1, padding=0):
        W = W * 2 -1
        return F.conv2d(X, W, stride=stride, padding=padding)

    def mzi_conv2d_function(self, X, W ,stride=1, padding=0):
        n_x = X.size(0)
        n_filters, d_filter, h_filter, w_filter = W.size()
        W_col, X_col, h_out, w_out = im2col_2d(W, X, stride, padding)
        if(self.mzi_noise is not None):
            pass



        out = W_col.matmul(X_col)**2

        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()
        return out

    def ring_conv2d_function(self, X, W, stride=1, padding=0):
        # ring can implement posneg weight
        n_x = X.size(0)
        n_filters, d_filter, h_filter, w_filter = W.size()
        W_col, X_col, h_out, w_out = im2col_2d(W, X*X, stride, padding)
        if(self.ring_noise is not None):
            W_col += self.ring_noise
        W_col = W_col.clamp(0, 1)
        if(self.ring_crosstalk_perc > 1e-6):
            kernel = torch.from_numpy(np.array([[0, self.ring_crosstalk_perc, 0],[self.ring_crosstalk_perc,0,self.ring_crosstalk_perc],[0, self.ring_crosstalk_perc, 0]])).float().to(self.device)
            W_col += F.conv2d(W_col.unsqueeze(0).unsqueeze(0).abs(), kernel.unsqueeze(0).unsqueeze(0), bias=None, padding=1, stride=1).squeeze()
        W_col = W_col.clamp(0, 1)
        # W represents energy transmission factor a, transfer function (2a-1)x**2
        W = 2*W - 1

        out = W_col.matmul(X_col)

        out = out.view(n_filters, h_out, w_out, n_x)
        out = out.permute(3, 0, 1, 2).contiguous()
        return out

    def forward(self, x):
        # if(self.mode=="oadder" and self.w_bit == 1):
        #     x_mean = x.data.mean(dim=1, keepdim=True)**2 # [bs, 1, h, w]
        #     w_mean = self.weight.data.mean(dim=[1,2,3]).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)**2 # [1, outc, 1, 1]
        #     coeff = x_mean * w_mean
        # print(x.mean().data.item(), x.std().data.item(), x.max().data.item())
        input = x.clamp(0,1)
        x = self.input_quantizer(x)
        if(self.input_augment and self.beta.data.max() > 1e-6):
            x = (1-self.beta) * x + self.beta * input
        weight = self.weight_quantizer(self.weight)


        output = self.adder_func(x, weight, self.stride, self.padding)
        # if(self.mode=="oadder" and self.w_bit == 1):
        #     x = x * coeff
        # output[self.output_channel // 2:] = output[self.output_channel // 2:] * -1

        if self.bias:
            output += self.b.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # factor = weight.view(self.output_channel, -1).norm(p=2, dim=1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        # output = output / factor / 1.21
        # output = self.alpha * input + output


        return output


class NormProp(nn.Module):
    def __init__(self, out_channels, act):
        super().__init__()
        self.mul = 1/np.sqrt(0.5*(1-1/np.pi))
        self.add = np.sqrt(1/2*np.pi)
        self.act = act
        self.out_channels = out_channels
        self.gamma = nn.Parameter(torch.zeros(out_channels))
        self.beta = nn.Parameter(torch.zeros(out_channels))
        nn.init.constant_(self.gamma, 1/1.21)
        nn.init.constant_(self.beta, 0)


    def forward(self, x, w):
        if(w.dim() == 2):
            w = w.view(w.size(0), -1).norm(p=2, dim=1).data
            gamma = self.gamma.unsqueeze(0)
            beta = self.beta.unsqueeze(0)
        else:
            w = w.view(w.size(0), -1).norm(p=2, dim=1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).data
            gamma = self.gamma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            beta = self.beta.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        x = self.mul*(self.act(gamma * x / w + beta) - self.add)
        return x




if __name__ == "__main__":
    device = torch.device("cuda")
    # set_torch_deterministic()
    # model1 = OAdder2d_Q(2,2,3, device=device, phase_noise_std=0.1)
    # avg_sin_phase = model1.phases.sin().mean()
    # set_torch_deterministic()
    # model2 = OAdder2d_Q(2,2,3, device=device, phase_noise_std=0)
    # x = torch.ones(2, 2, 4, 4, device=device)
    # print(model1(x))
    # print(model1(x)/avg_sin_phase)
    # print(model2(x))
    set_torch_deterministic()
    model = OAdder2d_Q(2,
                        2,
                        3,
                        stride=1,
                        padding=0,
                        in_bit=8,
                        w_bit=8,
                        mode="oadder",
                        bias=False,
                        device=device
                    )
    x = torch.randn(2, 2, 4, 4, device=device).clamp(0, 1)
    y = model(x)

    model.assign_engines(2,2,0.1,0.05,True)

    # model.enable_calibration()
    # model.static_pre_calibration()
    # y1 = model(x)
    model.disable_calibration()
    y2 = model(x)
    print(y)
    # print(y1)
    print(y2)
    # print(F.mse_loss(y,y1).data.item())
    print(F.mse_loss(y,y2).data.item())
    exit(1)

    set_torch_deterministic()
    model = OLinear_Q(8,
                    8,
                    in_bit=8,
                    w_bit=8,
                    bias=False,
                    device=device
                    )
    x = torch.ones(4, 8, device=device)
    y = model(x)

    model.assign_engines(2,2,0.1,0.05,True)
    # model.deassign_engines()
    # model.enable_calibration()
    # model.static_pre_calibration()
    # y1 = model(x)
    model.disable_calibration()
    y2 = model(x)
    print(y)
    # print(y1)
    print(y2)
    # print(F.mse_loss(y,y1).data.item())
    print(F.mse_loss(y,y2).data.item())


