#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-12 19:54:05
@LastEditTime: 2019-07-28 00:32:45
'''

from multiprocessing.dummy import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from layers import *
try:
    from matrix_parametrization import *
except:
    print("No unitary parametrization module found")
from utils import *


class FCMLP(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.device = device
        self.fc1 = nn.Linear(n_feat, 400, bias=False)
        self.fc2 = nn.Linear(400, 10, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        # x = x[..., ::2, ::2].contiguous()
        x = x.view(-1, self.n_feat)
        x = torch.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        return F.log_softmax(self.fc2(x), dim=1)


class FCMLP_MZI_Q(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, hidden_list=[16], v_bit=4, v_pi=4.36, v_max=10.8, clamp_small_phase_lead_percentile=1, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.hidden_list = hidden_list
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.clamp_small_phase_lead_percentile = clamp_small_phase_lead_percentile
        self.device = device
        # self.fc1 = nn.Linear(n_feat, 16, bias=False)
        # self.fc2 = nn.Linear(16, 10, bias=False)
        # self.layers = {"fc1": self.fc1, "fc2": self.fc2}
        self.buildLayers(act="relu")
        self.weights_backup = {}
        self.voltage_masks = {}
        self.voltage_backups = {}
        self.lipschitz_loss_dc = 0
        self.cross_layer_lipschitz_loss_dc = 0
        self.init_weights(initializer=nn.init.orthogonal_)
        self.init_voltage_mask_and_backup()
        self.getLipschitzLoss_DC()
        self.getCrossLayerLipschitzLoss_DC()
        self.pool = Pool(min(12, len(hidden_list)+1))
        # self.decomposer = RealUniaryDecomposer()
        # self.phase_quantize_fn = phase_quantize_fn_cpu(p_bit=4)

    def buildLayers(self, act="relun", act_clip_thres=4):
        self.layers = OrderedDict()
        self.acts = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            act_name = "act" + str(idx+1)
            in_channel = self.n_feat if idx == 0 else self.hidden_list[idx-1]
            out_channel = hidden_dim

            fc = nn.Linear(in_channel, out_channel, bias=False)
            self.layers[layer_name] = fc
            activation = nn.ReLU() if act == "relu" else ReLUN(act_clip_thres)
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            super().__setattr__(act_name, activation)
        layer_name = "fc"+str(len(self.hidden_list)+1)

        fc = nn.Linear(self.hidden_list[-1], self.n_class, bias=False)
        super().__setattr__(layer_name, fc)
        self.layers[layer_name] = fc
        # self.linears = nn.ModuleList(self.layers.values())

    def init_weights(self, initializer=nn.init.kaiming_normal_):
        for layer in self.layers:
            initializer(self.layers[layer].weight)

    def init_voltage_mask_and_backup(self):
        for layer in self.layers:
            M, N = self.layers[layer].weight.size()
            self.voltage_masks[layer] = {"U": np.zeros([M*(M-1)//2], dtype=np.bool),
                                         "V": np.zeros([N*(N-1)//2], dtype=np.bool)}
            self.voltage_backups[layer] = {"U": np.zeros([M*(M-1)//2], dtype=np.float64),
                                           "V": np.zeros([N*(N-1)//2], dtype=np.float64)}

    def getWeights(self):
        return {layer: self.layers[layer].weight for layer in self.layers}

    def getLipschitzLoss_DC(self):
        self.backupWeights()
        self.init_weights(initializer=nn.init.orthogonal_)
        self.lipschitz_loss_dc = self.getLipschitzLoss().data
        self.restoreWeights()

    def getLipschitzLoss(self):
        diff_list = []
        for layer in self.layers:
            W = self.layers[layer].weight
            W_sq = torch.mm(W.t(), W)
            diff = W_sq - torch.eye(n=W_sq.size(0),
                                    dtype=W_sq.dtype, device=W_sq.device)
            diff_list.append(diff.contiguous().view(-1))
            # diff_sq_list.append(diff.abs().contiguous().view(-1))
        diff = torch.cat(diff_list, dim=0)
        diff_sq = diff * diff
        loss = diff_sq.mean() - self.lipschitz_loss_dc

        return loss

    def getCrossLayerLipschitzLoss_DC(self):
        self.backupWeights()
        self.init_weights(initializer=nn.init.orthogonal_)
        self.cross_layer_lipschitz_loss_dc = self.getCrossLayerLipschitzLoss().data
        self.restoreWeights()

    def getCrossLayerLipschitzLoss(self):
        W_prod = None
        for layer in self.layers:
            W = self.layers[layer].weight
            W_prod = W if W_prod is None else torch.mm(W, W_prod)
        W_prod_sq = torch.mm(W_prod.t(), W_prod)
        diff = W_prod_sq - \
            torch.eye(n=W_prod_sq.size(0), dtype=W_prod_sq.dtype,
                      device=W_prod_sq.device)
        diff_sq = diff * diff
        loss = diff_sq.mean() - self.cross_layer_lipschitz_loss_dc
        return loss

    def getQuantizationLoss(self, hessian_dict=None):
        diff_list = []
        for layer in self.layers:
            W = self.layers[layer].weight
            W_q = quantize_voltage_of_matrix_cpu(W.data, v_bit=self.v_bit, v_pi=self.v_pi, v_max=self.v_max,
                                                 clamp_small_phase_lead_percentile=self.clamp_small_phase_lead_percentile, output_device=self.device)
            diff = W - W_q
            diff_list.append(diff.contiguous().view(-1))
        diff = torch.cat(diff_list, dim=0)
        loss = (diff * diff).mean()

        # if(hessian_dict is not None):
        #     hessian_list = []
        #     for layer in self.layers:
        #         hessian = hessian_dict[layer]
        #         hessian_list.append(hessian.contiguous().view(-1))
        #     hessian = torch.cat(hessian_list, dim=0)
        #     loss = (torch.sqrt(hessian) * diff).mean()**2
        # else:
        #     loss = (diff * diff).mean()

        return loss

    def backupWeights(self):
        for layer in self.layers:
            self.weights_backup[layer] = torch.clone(
                self.layers[layer].weight.data)
        print(f"[I] Weights backuped")

    def restoreWeights(self):
        for layer in self.layers:
            assert layer in self.weights_backup and isinstance(
                self.weights_backup[layer], torch.Tensor), "[E] Weight restore failed. Please call backupWeights before restoreWeights"
            self.layers[layer].weight.data.copy_(self.weights_backup[layer])
        print(f"[I] Weights restored from backup")

    def applyVoltageQuantization_kernel(self, W, voltage_mask_U=None, voltage_backup_U=None, voltage_mask_V=None, voltage_backup_V=None, quantize_voltage_percentile=1):
        W_q = quantize_voltage_of_matrix_cpu(
            W,
            v_bit=self.v_bit,
            v_pi=self.v_pi,
            v_max=self.v_max,
            voltage_mask_U=voltage_mask_U,
            voltage_backup_U=voltage_backup_U,
            voltage_mask_V=voltage_mask_V,
            voltage_backup_V=voltage_backup_V,
            quantize_voltage_percentile=quantize_voltage_percentile,
            output_device=self.device
        )
        W.data.copy_(W_q)

    def applyVoltageQuantization(self, quantize_voltage_percentile=1, with_mask=False):
        assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0,1]"
        for layer in self.layers:
            if(with_mask == True):
                self.applyVoltageQuantization_kernel(
                    self.layers[layer].weight,
                    self.voltage_masks[layer]["U"],
                    self.voltage_backups[layer]["U"],
                    self.voltage_masks[layer]["V"],
                    self.voltage_backups[layer]["V"],
                    quantize_voltage_percentile=quantize_voltage_percentile
                )
            else:
                self.applyVoltageQuantization_kernel(
                    self.layers[layer].weight,
                    None, None, None, None,
                    quantize_voltage_percentile=quantize_voltage_percentile
                )
        print("[I] Quantization applied")

    def applyVoltageMask(self):
        for layer in self.layers:
            maintain_quantized_voltage_cpu(
                self.layers[layer].weight,
                self.v_pi,
                self.voltage_masks[layer]["U"],
                self.voltage_backups[layer]["U"],
                self.voltage_masks[layer]["V"],
                self.voltage_backups[layer]["V"],
                output_device=self.device
            )
            # self.layers[layer].weight.copy_(W_maintain)

    def applyUnitaryProjection_MT(self):
        tasks = []
        for layer in self.layers:
            tasks.append(self.layers[layer].weight)
        self.pool.map(lambda U: U.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(U.data.detach().cpu().numpy())).to(self.device)),
                      tasks)

    def forward(self, x):
        x = x.view(-1, self.n_feat)
        n_layer = len(self.layers)
        for idx, layer in enumerate(self.layers):
            fc = self.layers[layer]
            x = fc(x)
            if(idx + 1 < n_layer):
                act = self.acts[layer]
                x = act(x)
        # x = torch.relu(self.layers["fc1"](x))
        # return F.log_softmax(self.layers["fc2"](x), dim=1)
        return F.log_softmax(x, dim=1)


class FCMLP_Dorefa(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, hidden_list=[16], v_bit=4, v_pi=4.36, v_max=10.8, clamp_small_phase_lead_percentile=1, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.hidden_list = hidden_list
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.clamp_small_phase_lead_percentile = clamp_small_phase_lead_percentile
        self.device = device
        # self.fc1 = nn.Linear(n_feat, 16, bias=False)
        # self.fc2 = nn.Linear(16, 10, bias=False)
        # self.layers = {"fc1": self.fc1, "fc2": self.fc2}
        self.buildLayers(act="relun")
        self.weights_backup = {}
        self.voltage_masks = {}
        self.voltage_backups = {}
        self.lipschitz_loss_dc = 0
        self.cross_layer_lipschitz_loss_dc = 0
        self.init_weights(initializer=nn.init.orthogonal_)
        # self.init_weights()
        self.init_voltage_mask_and_backup()
        self.getLipschitzLoss_DC()
        self.getCrossLayerLipschitzLoss_DC()
        self.pool = Pool(min(12, len(hidden_list)+1))
        # self.decomposer = RealUniaryDecomposer()
        # self.phase_quantize_fn = phase_quantize_fn_cpu(p_bit=4)

    def buildLayers(self, act="relun", act_clip_thres=4):
        # assert act in {"relu", "relun"}, f"[E] Not supported activation function: {act}"
        # assert act_clip_thres > 0, f"[E] Threshold {act_clip_thres} is not supported by ReLUN()"
        self.layers = OrderedDict()
        self.acts = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            in_channel = self.n_feat if idx == 0 else self.hidden_list[idx-1]
            out_channel = hidden_dim
            # fc = BlockUSVLinear(in_channel, out_channel, self.block_list[idx],
            #                     use_bias=False, S_trainable=self.S_trainable)
            fc = linear_Q_fn(self.v_bit)(in_channel, out_channel, bias=False)
            self.layers[layer_name] = fc
            activation = nn.ReLU() if act == "relu" else ReLUN(act_clip_thres)
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            super().__setattr__(layer_name, activation)
        layer_name = "fc"+str(len(self.hidden_list)+1)
        # fc = BlockUSVLinear(self.hidden_list[-1], self.n_class, self.block_list[-1],
        #                     use_bias=False, S_trainable=self.S_trainable)
        fc = linear_Q_fn(self.v_bit)(
            self.hidden_list[-1], self.n_class, bias=False)
        super().__setattr__(layer_name, fc)
        self.layers[layer_name] = fc
        self.linears = nn.ModuleList(self.layers.values())

    def init_weights(self, initializer=nn.init.kaiming_normal_):
        for layer in self.layers:
            initializer(self.layers[layer].weight)
            print(f"[I] mean: {self.layers[layer].weight.mean()} std: {self.layers[layer].weight.std()} max :{self.layers[layer].weight.max()}, min: {self.layers[layer].weight.min()}")

    def init_voltage_mask_and_backup(self):
        for layer in self.layers:
            M, N = self.layers[layer].weight.size()
            self.voltage_masks[layer] = {"U": np.zeros([M*(M-1)//2], dtype=np.bool),
                                         "V": np.zeros([N*(N-1)//2], dtype=np.bool)}
            self.voltage_backups[layer] = {"U": np.zeros([M*(M-1)//2], dtype=np.float64),
                                           "V": np.zeros([N*(N-1)//2], dtype=np.float64)}

    def getWeights(self):
        return {layer: self.layers[layer].weight for layer in self.layers}

    def getLipschitzLoss_DC(self):
        self.backupWeights()
        self.init_weights(initializer=nn.init.orthogonal_)
        self.lipschitz_loss_dc = self.getLipschitzLoss().data
        self.restoreWeights()

    def getLipschitzLoss(self):
        diff_list = []
        for layer in self.layers:
            W = self.layers[layer].weight
            W_sq = torch.mm(W.t(), W)
            diff = W_sq - torch.eye(n=W_sq.size(0),
                                    dtype=W_sq.dtype, device=W_sq.device)
            diff_list.append(diff.contiguous().view(-1))
            # diff_sq_list.append(diff.abs().contiguous().view(-1))
        diff = torch.cat(diff_list, dim=0)
        diff_sq = diff * diff
        loss = diff_sq.mean() - self.lipschitz_loss_dc

        return loss

    def getCrossLayerLipschitzLoss_DC(self):
        self.backupWeights()
        self.init_weights(initializer=nn.init.orthogonal_)
        self.cross_layer_lipschitz_loss_dc = self.getCrossLayerLipschitzLoss().data
        self.restoreWeights()

    def getCrossLayerLipschitzLoss(self):
        W_prod = None
        for layer in self.layers:
            W = self.layers[layer].weight
            W_prod = W if W_prod is None else torch.mm(W, W_prod)
        W_prod_sq = torch.mm(W_prod.t(), W_prod)
        diff = W_prod_sq - \
            torch.eye(n=W_prod_sq.size(0), dtype=W_prod_sq.dtype,
                      device=W_prod_sq.device)
        diff_sq = diff * diff
        loss = diff_sq.mean() - self.cross_layer_lipschitz_loss_dc
        return loss

    def getQuantizationLoss(self, hessian_dict=None):
        diff_list = []
        for layer in self.layers:
            W = self.layers[layer].weight
            W_q = quantize_voltage_of_matrix_cpu(W.data, v_bit=self.v_bit, v_pi=self.v_pi, v_max=self.v_max,
                                                 clamp_small_phase_lead_percentile=self.clamp_small_phase_lead_percentile, output_device=self.device)
            diff = W - W_q
            diff_list.append(diff.contiguous().view(-1))
        diff = torch.cat(diff_list, dim=0)
        loss = (diff * diff).mean()

        # if(hessian_dict is not None):
        #     hessian_list = []
        #     for layer in self.layers:
        #         hessian = hessian_dict[layer]
        #         hessian_list.append(hessian.contiguous().view(-1))
        #     hessian = torch.cat(hessian_list, dim=0)
        #     loss = (torch.sqrt(hessian) * diff).mean()**2
        # else:
        #     loss = (diff * diff).mean()

        return loss

    def backupWeights(self):
        for layer in self.layers:
            self.weights_backup[layer] = torch.clone(
                self.layers[layer].weight.data)
        print(f"[I] Weights backuped")

    def restoreWeights(self):
        for layer in self.layers:
            assert layer in self.weights_backup and isinstance(
                self.weights_backup[layer], torch.Tensor), "[E] Weight restore failed. Please call backupWeights before restoreWeights"
            self.layers[layer].weight.data.copy_(self.weights_backup[layer])
        print(f"[I] Weights restored from backup")

    def applyVoltageQuantization_kernel(self, W, voltage_mask_U=None, voltage_backup_U=None, voltage_mask_V=None, voltage_backup_V=None, quantize_voltage_percentile=1):
        W_q = quantize_voltage_of_matrix_cpu(
            W,
            v_bit=self.v_bit,
            v_pi=self.v_pi,
            v_max=self.v_max,
            voltage_mask_U=voltage_mask_U,
            voltage_backup_U=voltage_backup_U,
            voltage_mask_V=voltage_mask_V,
            voltage_backup_V=voltage_backup_V,
            quantize_voltage_percentile=quantize_voltage_percentile,
            output_device=self.device
        )
        W.data.copy_(W_q)

    def applyVoltageQuantization(self, quantize_voltage_percentile=1, with_mask=False):
        assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0,1]"
        for layer in self.layers:
            if(with_mask == True):
                self.applyVoltageQuantization_kernel(
                    self.layers[layer].weight,
                    self.voltage_masks[layer]["U"],
                    self.voltage_backups[layer]["U"],
                    self.voltage_masks[layer]["V"],
                    self.voltage_backups[layer]["V"],
                    quantize_voltage_percentile=quantize_voltage_percentile
                )
            else:
                self.applyVoltageQuantization_kernel(
                    self.layers[layer].weight,
                    None, None, None, None,
                    quantize_voltage_percentile=quantize_voltage_percentile
                )
        print("[I] Quantization applied")

    def applyVoltageMask(self):
        for layer in self.layers:
            maintain_quantized_voltage_cpu(
                self.layers[layer].weight,
                self.v_pi,
                self.voltage_masks[layer]["U"],
                self.voltage_backups[layer]["U"],
                self.voltage_masks[layer]["V"],
                self.voltage_backups[layer]["V"],
                output_device=self.device
            )
            # self.layers[layer].weight.copy_(W_maintain)

    def applyUnitaryProjection_MT(self):
        tasks = []
        for layer in self.layers:
            tasks.append(self.layers[layer].weight)
        self.pool.map(lambda U: U.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(U.data.detach().cpu().numpy())).to(self.device)),
                      tasks)

    def forward(self, x):
        # x = x[..., ::4, ::4].contiguous()
        # x = F.avg_pool2d(x, kernel_size=4, stride=4, padding=0).contiguous()
        x = x.view(-1, self.n_feat)
        n_layer = len(self.layers)
        for idx, layer in enumerate(self.layers):
            fc = self.layers[layer]
            x = fc(x)
            if(idx + 1 < n_layer):
                act = self.acts[layer]
                x = act(x)
        # x = torch.relu(self.layers["fc1"](x))
        # return F.log_softmax(self.layers["fc2"](x), dim=1)
        return F.log_softmax(x, dim=1)


class FCMLP_USV_Q(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, hidden_list=[16], S_trainable=True, v_bit=4, v_pi=4.36, v_max=10.8, clamp_small_phase_lead_percentile=1, n_thread=4, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.hidden_list = hidden_list
        self.S_trainable = S_trainable
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.clamp_small_phase_lead_percentile = clamp_small_phase_lead_percentile
        self.device = device
        # self.fc1 = USVLinear(n_feat, 16, use_bias=False, device=self.device)
        # self.fc2 = USVLinear(16, 10, use_bias=False, device=self.device)
        self.buildLayers()
        # self.act1 = ReLUN(4)
        # self.act1 = nn.ReLU()
        # self.layers = {"fc1": self.fc1, "fc2": self.fc2}
        self.weights_backup = {}
        self.add_weight_backup(cache_name="val_bk")
        # self.weights_backup = {'val_bk': {"fc1":{}, "fc2":{}}}
        self.voltage_masks = {}
        self.voltage_backups = {}
        self.lipschitz_loss_dc = 0
        self.cross_layer_lipschitz_loss_dc = 0
        self.init_weights(initializer=nn.init.orthogonal_)
        self.init_voltage_mask_and_backup()
        self.getLipschitzLoss_DC()
        self.getCrossLayerLipschitzLoss_DC()
        self.pool = Pool(n_thread)
        # self.decomposer = RealUniaryDecomposer()
        # self.phase_quantize_fn = phase_quantize_fn_cpu(p_bit=4)

    def buildLayers(self, act="relun", act_clip_thres=4):
        self.layers = OrderedDict()
        self.acts = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            act_name = "act" + str(idx+1)
            in_channel = self.n_feat if idx == 0 else self.hidden_list[idx-1]
            out_channel = hidden_dim

            fc = USVLinear(in_channel, out_channel,
                           use_bias=False, S_trainable=self.S_trainable)
            activation = nn.ReLU() if act == "relu" else ReLUN(act_clip_thres)
            self.layers[layer_name] = fc
            self.acts[layer_name] = activation

            super().__setattr__(layer_name, fc)
            super().__setattr__(act_name, activation)

        layer_name = "fc"+str(len(self.hidden_list)+1)

        fc = USVLinear(self.hidden_list[-1], self.n_class,
                       use_bias=False, S_trainable=self.S_trainable)
        super().__setattr__(layer_name, fc)
        self.layers[layer_name] = fc

        # self.linears = nn.ModuleList(self.layers.values())

    def init_weights(self, initializer=nn.init.kaiming_normal_):
        for layer in self.layers:
            # print(self.layers[layer])
            self.layers[layer].init_weights()

    def init_weights_from_full_precision_model(self, model):
        for layer in model.layers:
            W = model.layers[layer].weight.data.detach().cpu().numpy()
            # print(W)
            U, S, V = np.linalg.svd(W, full_matrices=True)
            self.layers[layer].U.data.copy_(
                torch.from_numpy(U).to(torch.float32).to(self.device))
            self.layers[layer].S.data.copy_(
                torch.from_numpy(S).to(torch.float32).to(self.device))
            self.layers[layer].V.data.copy_(
                torch.from_numpy(V).to(torch.float32).to(self.device))
            # W_recon = self.layers[layer].buildWeight_from_USV(self.layers[layer].U, self.layers[layer].S,self.layers[layer].V)
            # print(W_recon)
        print("[I] Initialize from full precision model")

    def add_weight_backup(self, cache_name):
        if(cache_name in self.weights_backup):
            print(f"[I] {cache_name} already exists, skip creation")
            return
        self.weights_backup[cache_name] = {layer: {} for layer in self.layers}
        print(f"[I] New cache {cache_name} created")

    def init_voltage_mask_and_backup(self):
        for layer in self.layers:
            M, N = self.layers[layer].U.size(0), self.layers[layer].V.size(0)
            self.voltage_masks[layer] = {"U": np.zeros([M*(M-1)//2], dtype=np.bool),
                                         "V": np.zeros([N*(N-1)//2], dtype=np.bool)}
            self.voltage_backups[layer] = {"U": np.zeros([M*(M-1)//2], dtype=np.float64),
                                           "V": np.zeros([N*(N-1)//2], dtype=np.float64)}

    def getUSV(self):
        return {layer: {"U": self.layers[layer].U, "S": self.layers[layer].S, "V": self.layers[layer].V} for layer in self.layers}

    def getLipschitzLoss_DC(self):
        self.backupWeights()
        self.init_weights(initializer=nn.init.orthogonal_)
        self.lipschitz_loss_dc = self.getLipschitzLoss().data
        self.restoreWeights()

    def getLipschitzLoss(self):
        diff_list = []
        for layer in self.layers:
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            W = self.layers[layer].buildWeight_from_USV(U, S, V)
            W_sq = torch.mm(W.t(), W)
            diff = W_sq - torch.eye(n=W_sq.size(0),
                                    dtype=W_sq.dtype, device=W_sq.device)
            diff_list.append(diff.contiguous().view(-1))
            # diff_sq_list.append(diff.abs().contiguous().view(-1))
        diff = torch.cat(diff_list, dim=0)
        diff_sq = diff * diff
        loss = diff_sq.mean() - self.lipschitz_loss_dc

        return loss

    def getSigmaRegLoss(self):
        if(self.S_trainable):
            s_list = []
            for layer in self.layers:
                s_list.append(self.layers[layer].S.contiguous().view(-1))
            s_list = torch.cat(s_list, dim=0)
            loss = torch.mean(s_list * s_list)

    def getCrossLayerLipschitzLoss_DC(self):
        self.backupWeights()
        self.init_weights(initializer=nn.init.orthogonal_)
        self.cross_layer_lipschitz_loss_dc = self.getCrossLayerLipschitzLoss().data
        self.restoreWeights()

    def getCrossLayerLipschitzLoss(self):
        W_prod = None
        for layer in self.layers:
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            W = self.layers[layer].buildWeight_from_USV(U, S, V)
            W_prod = W if W_prod is None else torch.mm(W, W_prod)
        W_prod_sq = torch.mm(W_prod.t(), W_prod)
        diff = W_prod_sq - \
            torch.eye(n=W_prod_sq.size(0), dtype=W_prod_sq.dtype,
                      device=W_prod_sq.device)
        diff_sq = diff * diff
        loss = diff_sq.mean() - self.cross_layer_lipschitz_loss_dc
        return loss

    def getQuantizationLoss(self, quantize_voltage_percentile=1, hessian_dict=None):
        diff_list = []
        for layer in self.layers:
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            W = self.layers[layer].buildWeight_from_USV(U, S, V)
            U_q = quantize_voltage_of_unitary_cpu(
                U,
                v_bit=self.v_bit,
                v_pi=self.v_pi,
                v_max=self.v_max,
                voltage_mask=self.voltage_masks[layer]["U"],
                voltage_backup=self.voltage_backups[layer]["U"],
                quantize_voltage_percentile=quantize_voltage_percentile,
                output_device=self.device
            )
            V_q = quantize_voltage_of_unitary_cpu(
                V,
                v_bit=self.v_bit,
                v_pi=self.v_pi,
                v_max=self.v_max,
                voltage_mask=self.voltage_masks[layer]["V"],
                voltage_backup=self.voltage_backups[layer]["V"],
                quantize_voltage_percentile=quantize_voltage_percentile,
                output_device=self.device
            )
            W_q = self.layers[layer].buildWeight_from_USV(
                U_q.data, S.data, V_q.data).data

            diff = W - W_q.data
            diff_list.append(diff.contiguous().view(-1))
        diff = torch.cat(diff_list, dim=0)
        loss = (diff * diff).mean()

        return loss

    def backupWeights(self, cache_name="val_bk"):
        if(cache_name not in self.weights_backup):
            self.add_weight_backup(cache_name=cache_name)
        for layer in self.layers:
            self.weights_backup[cache_name][layer]["U"] = torch.clone(
                self.layers[layer].U.data)
            self.weights_backup[cache_name][layer]["S"] = torch.clone(
                self.layers[layer].S.data)
            self.weights_backup[cache_name][layer]["V"] = torch.clone(
                self.layers[layer].V.data)
        # print(f"[I] Weights {cache_name} backuped")

    def restoreWeights(self, cache_name='val_bk'):
        if(cache_name not in self.weights_backup):
            print(
                f"[W] No cache named {cache_name} to restore, skip restoring")
            return False
        for layer in self.layers:
            # assert layer in self.weights_backup[cache_name] and isinstance(self.weights_backup[cache_name][layer]["U"], torch.Tensor) and isinstance(self.weights_backup[cache_name][layer]["S"], torch.Tensor) and isinstance(self.weights_backup[cache_name][layer]["V"], torch.Tensor), "[E] Weight restore failed. Please call backupWeights before restoreWeights"
            if("U" in self.weights_backup[cache_name][layer]):
                self.layers[layer].U.data.copy_(
                    self.weights_backup[cache_name][layer]["U"])
            if("S" in self.weights_backup[cache_name][layer]):
                self.layers[layer].S.data.copy_(
                    self.weights_backup[cache_name][layer]["S"])
            if("V" in self.weights_backup[cache_name][layer]):
                self.layers[layer].V.data.copy_(
                    self.weights_backup[cache_name][layer]["V"])
        # print(f"[I] Weights {cache_name} restored from backup")
        return True

    def applyVoltageQuantization_kernel(self, U, V, voltage_mask_U=None, voltage_backup_U=None, voltage_mask_V=None, voltage_backup_V=None, quantize_voltage_percentile=1, strict_mask=True):
        U_q = quantize_voltage_of_unitary_cpu(
            U,
            v_bit=self.v_bit,
            v_pi=self.v_pi,
            v_max=self.v_max,
            voltage_mask=voltage_mask_U,
            voltage_backup=voltage_backup_U,
            quantize_voltage_percentile=quantize_voltage_percentile,
            strict_mask=strict_mask,
            output_device=self.device
        )
        V_q = quantize_voltage_of_unitary_cpu(
            V,
            v_bit=self.v_bit,
            v_pi=self.v_pi,
            v_max=self.v_max,
            voltage_mask=voltage_mask_V,
            voltage_backup=voltage_backup_V,
            quantize_voltage_percentile=quantize_voltage_percentile,
            strict_mask=strict_mask,
            output_device=self.device
        )
        U.data.copy_(U_q)
        V.data.copy_(V_q)

    def applyVoltageQuantization(self, quantize_voltage_percentile=1, with_mask=False, strict_mask=True):
        assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0,1]"
        for layer in self.layers:
            if(with_mask == True):
                self.applyVoltageQuantization_kernel(
                    self.layers[layer].U,
                    self.layers[layer].V,
                    self.voltage_masks[layer]["U"],
                    self.voltage_backups[layer]["U"],
                    self.voltage_masks[layer]["V"],
                    self.voltage_backups[layer]["V"],
                    quantize_voltage_percentile=quantize_voltage_percentile,
                    strict_mask=strict_mask
                )
            else:
                self.applyVoltageQuantization_kernel(
                    self.layers[layer].U,
                    self.layers[layer].V,
                    None, None, None, None,
                    quantize_voltage_percentile=quantize_voltage_percentile,
                    strict_mask=strict_mask
                )
        print("[I] Quantization applied")

    def applyVoltageQuantization_MT(self, quantize_voltage_percentile=1, with_mask=False, n_thread=8, strict_mask=True):
        assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0,1]"
        tasks = []
        for layer in self.layers:
            if(with_mask == True):
                tasks.append(
                    (self.layers[layer].U, self.voltage_masks[layer]["U"], self.voltage_backups[layer]["U"]))
                tasks.append(
                    (self.layers[layer].V, self.voltage_masks[layer]["V"], self.voltage_backups[layer]["V"]))
            else:
                tasks.append((self.layers[layer].U, None, None))
                tasks.append((self.layers[layer].V, None, None))
        self.pool.map(lambda args: args[0].data.copy_(quantize_voltage_of_unitary_cpu(
            args[0],
            v_bit=self.v_bit,
            v_pi=self.v_pi,
            v_max=self.v_max,
            voltage_mask=args[1],
            voltage_backup=args[2],
            quantize_voltage_percentile=quantize_voltage_percentile,
            strict_mask=strict_mask,
            output_device=self.device)),
            tasks)

    def applyVoltageMask(self, gamma_noise_std=0):
        for layer in self.layers:
            maintain_quantized_voltage_of_unitary_cpu(
                self.layers[layer].U,
                self.v_pi,
                self.voltage_masks[layer]["U"],
                self.voltage_backups[layer]["U"],
                gamma_noise_std=gamma_noise_std,
                output_device=self.device
            )
            maintain_quantized_voltage_of_unitary_cpu(
                self.layers[layer].V,
                self.v_pi,
                self.voltage_masks[layer]["V"],
                self.voltage_backups[layer]["V"],
                gamma_noise_std=gamma_noise_std,
                output_device=self.device
            )

    def applyVoltageMask_MT(self, gamma_noise_std=0, ori_cache_name=None, noise_cache_name=None, decay_rate=0, learning_rate=0, clip_voltage=False, thres_cache_name="clip_thres_bk"):
        tasks = []
        for layer in self.layers:
            tasks.append((self.layers[layer].U,
                          self.voltage_masks[layer]["U"],
                          self.voltage_backups[layer]["U"],
                          self.weights_backup[ori_cache_name][layer],
                          self.weights_backup[noise_cache_name][layer],
                          "U",
                          self.weights_backup[thres_cache_name][layer]["U"][0] if clip_voltage else float(
                              "-inf"),
                          self.weights_backup[thres_cache_name][layer]["U"][1] if clip_voltage else float(
                              "inf")
                          ))
            tasks.append((self.layers[layer].V,
                          self.voltage_masks[layer]["V"],
                          self.voltage_backups[layer]["V"],
                          self.weights_backup[ori_cache_name][layer],
                          self.weights_backup[noise_cache_name][layer],
                          "V",
                          self.weights_backup[thres_cache_name][layer]["V"][0] if clip_voltage else float(
                              "-inf"),
                          self.weights_backup[thres_cache_name][layer]["V"][1] if clip_voltage else float(
                              "inf")
                          ))

        def func(U, voltage_masks, voltage_backup, ori_weight_backup_dict, noise_weight_backup_dict, weight_backup_key, lower_thres, upper_thres):
            U_recon, U_recon_n = maintain_quantized_voltage_of_unitary_cpu(
                U,
                self.v_pi,
                voltage_masks,
                voltage_backup,
                gamma_noise_std=gamma_noise_std,
                weight_decay_rate=decay_rate,
                learning_rate=learning_rate,
                clip_voltage=clip_voltage,
                lower_thres=lower_thres,
                upper_thres=upper_thres,
                output_device=self.device
            )
            ori_weight_backup_dict[weight_backup_key] = U_recon.data.clone()
            if(U_recon_n is not None):
                noise_weight_backup_dict[weight_backup_key] = U_recon_n.data.clone(
                )

        self.pool.map(lambda args: func(*args), tasks)

    def applyUnitaryProjection(self):
        for layer in self.layers:
            self.layers[layer].U.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(
                self.layers[layer].U.data.detach().cpu().numpy())).to(self.device))
            self.layers[layer].V.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(
                self.layers[layer].V.data.detach().cpu().numpy())).to(self.device))

    def applyUnitaryProjection_MT(self):
        tasks = []
        for layer in self.layers:
            tasks.append(self.layers[layer].U)
            tasks.append(self.layers[layer].V)
        self.pool.map(lambda U: U.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(U.data.detach().cpu().numpy())).to(self.device)),
                      tasks)

        # print("[I] Unitary projection done")

    def addGammaNoise(self, gamma_noise_std=0):
        for layer in self.layers:
            U, V = self.layers[layer].U, self.layers[layer].V
            U_n = add_gamma_noise_to_unitary_cpu(
                U, v_pi=self.v_pi, gamma_noise_std=gamma_noise_std, output_device=self.device)
            V_n = add_gamma_noise_to_unitary_cpu(
                V, v_pi=self.v_pi, gamma_noise_std=gamma_noise_std, output_device=self.device)
            U.data.copy_(U_n)
            V.data.copy_(V_n)

    def addPhaseNoise(self, phase_noise_std=0, protect_rate=0, protect_layer_list={}):
        for layer in self.layers:
            if(layer in protect_layer_list):
                continue
            U, V = self.layers[layer].U, self.layers[layer].V
            M, N = U.size(0), V.size(0)
            mask = np.random.choice(a=[False, True], size=[
                                    M*(M-1)//2], p=[1-protect_rate, protect_rate])
            U_n = add_phase_noise_to_unitary_cpu(
                U, phase_noise_std=phase_noise_std, protect_mask=mask, output_device=self.device)
            mask = np.random.choice(a=[False, True], size=[
                                    N*(N-1)//2], p=[1-protect_rate, protect_rate])
            V_n = add_phase_noise_to_unitary_cpu(
                V, phase_noise_std=phase_noise_std, protect_mask=mask, output_device=self.device)
            U.data.copy_(U_n)
            V.data.copy_(V_n)

    def calcClipVoltageThreshold(self, lower_perc=0, upper_perc=1, cache_name="clip_thres_bk"):

        if(cache_name not in self.weights_backup):
            self.add_weight_backup(cache_name)
        decomposer = RealUnitaryDecomposer()
        gamma = np.pi / (self.v_pi**2)
        quantizer = voltage_quantize_fn_cpu(
            v_bit=self.v_bit, v_pi=self.v_pi, v_max=self.v_max)
        for layer in self.layers:
            U, V = self.layers[layer].U, self.layers[layer].V
            W_numpy = U.detach().cpu().numpy().copy().astype(np.float64)

            N = W_numpy.shape[0]
            delta_list, phi_mat = decomposer.decompose(W_numpy)
            phi_list = upper_triangle_to_vector_cpu(phi_mat)
            v_list = phase_to_voltage_cpu(phi_list, gamma)

            lower_thres = quantizer(np.percentile(v_list, lower_perc * 100))
            upper_thres = quantizer(np.percentile(v_list, upper_perc * 100))

            self.weights_backup[cache_name][layer]["U"] = (
                lower_thres, upper_thres)

            W_numpy = V.detach().cpu().numpy().copy().astype(np.float64)
            N = W_numpy.shape[0]
            delta_list, phi_mat = decomposer.decompose(W_numpy)
            phi_list = upper_triangle_to_vector_cpu(phi_mat)
            v_list = phase_to_voltage_cpu(phi_list, gamma)

            lower_thres = quantizer(np.percentile(v_list, lower_perc * 100))
            upper_thres = quantizer(np.percentile(v_list, upper_perc * 100))
            self.weights_backup[cache_name][layer]["V"] = (
                lower_thres, upper_thres)

        print("[I] Voltage clipping thresholds are cached")
        print(self.weights_backup[cache_name])

    def initClipThresScheduler(self, step_beg, step_end, beg_cache="clip_thres_beg_bk", end_cache="clip_thres_end_bk"):
        thres_cache = "clip_thres_scheduler_bk"
        self.add_weight_backup(thres_cache)
        for layer in self.layers:

            lower_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["U"][0], self.weights_backup[end_cache][layer]["U"][0])
            upper_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["U"][1], self.weights_backup[end_cache][layer]["U"][1])
            self.weights_backup[thres_cache][layer]["U"] = (
                lower_scheduler, upper_scheduler)

            lower_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["V"][0], self.weights_backup[end_cache][layer]["V"][0])
            upper_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["V"][1], self.weights_backup[end_cache][layer]["V"][1])
            self.weights_backup[thres_cache][layer]["V"] = (
                lower_scheduler, upper_scheduler)

    def updateClipVoltageThreshold(self, epoch, thres_cache="clip_thres_bk"):
        scheduler_cache = "clip_thres_scheduler_bk"
        for layer in self.layers:
            lower_scheduler = self.weights_backup[scheduler_cache][layer]["U"][0]
            upper_scheduler = self.weights_backup[scheduler_cache][layer]["U"][1]
            self.weights_backup[thres_cache][layer]["U"] = (
                lower_scheduler(epoch), upper_scheduler(epoch))

            lower_scheduler = self.weights_backup[scheduler_cache][layer]["V"][0]
            upper_scheduler = self.weights_backup[scheduler_cache][layer]["V"][1]
            self.weights_backup[thres_cache][layer]["V"] = (
                lower_scheduler(epoch), upper_scheduler(epoch))

    def clipVoltages_MT(self, cache_name="clip_thres_bk"):
        tasks = []
        for layer in self.layers:
            U, V = self.layers[layer].U, self.layers[layer].V
            thres_U = self.weights_backup[cache_name][layer]["U"]
            thres_V = self.weights_backup[cache_name][layer]["V"]
            tasks.append([U, thres_U[0], thres_U[1]])
            tasks.append([V, thres_V[0], thres_V[1]])

        self.pool.map(lambda args: args[0].data.copy_(clip_voltage_of_unitary_cpu(
            args[0], v_pi=self.v_pi, lower_thres=args[1], upper_thres=args[2], output_device=self.device)), tasks)

        # print(f"[I] Voltages are pruned, ({lower_perc}, {upper_perc})")

    def forward(self, x):
        # x = x[..., ::4, ::4].contiguous()
        # x = F.avg_pool2d(x, kernel_size=4, stride=4, padding=0).contiguous()
        # x = x.view(-1, self.n_feat)
        # x = self.fc1(x)
        # x = x.clamp(0, 4)
        # x = self.act1(x)
        # return F.log_softmax(self.fc2(x), dim=1)
        x = x.view(-1, self.n_feat)

        n_layer = len(self.layers)
        for idx, layer in enumerate(self.layers):
            fc = self.layers[layer]
            x = fc(x)
            if(idx + 1 < n_layer):
                act = self.acts[layer]
                x = act(x)
        # x = torch.relu(self.layers["fc1"](x))
        # return F.log_softmax(self.layers["fc2"](x), dim=1)
        out = F.log_softmax(x, dim=1)
        # print(out.size())
        return out


class FCMLP_BlockUSV_Q(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, hidden_list=[16], block_list=[8, 10], S_trainable=True, v_bit=4, v_pi=4.36, v_max=10.8, clamp_small_phase_lead_percentile=1, n_thread=4, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.hidden_list = hidden_list
        self.block_list = block_list
        self.S_trainable = S_trainable
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.clamp_small_phase_lead_percentile = clamp_small_phase_lead_percentile
        self.device = device
        self.buildLayers(act="relu")
        # self.act1 = ReLUN(4)
        # self.act1 = nn.ReLU()
        # self.layers = {"fc1": self.fc1, "fc2": self.fc2}
        self.weights_backup = {}
        self.add_weight_backup(cache_name="val_bk")
        # self.weights_backup = {'val_bk': {"fc1":{}, "fc2":{}}}
        self.voltage_masks = {}
        self.voltage_backups = {}
        self.lipschitz_loss_dc = 0
        self.cross_layer_lipschitz_loss_dc = 0
        self.init_weights(initializer=nn.init.orthogonal_)
        self.init_voltage_mask_and_backup()
        # self.getLipschitzLoss_DC()
        # self.getCrossLayerLipschitzLoss_DC()
        self.pool = Pool(n_thread)
        # self.decomposer = RealUniaryDecomposer()
        # self.phase_quantize_fn = phase_quantize_fn_cpu(p_bit=4)

    def buildLayers(self, act="relun", act_clip_thres=4):
        # assert act in {"relu", "relun"}, f"[E] Not supported activation function: {act}"
        # assert act_clip_thres > 0, f"[E] Threshold {act_clip_thres} is not supported by ReLUN()"
        self.layers = OrderedDict()
        self.acts = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            act_name = "act" + str(idx+1)
            in_channel = self.n_feat if idx == 0 else self.hidden_list[idx-1]
            out_channel = hidden_dim
            fc = BlockUSVLinear(in_channel, out_channel, self.block_list[idx],
                                use_bias=False, S_trainable=self.S_trainable)
            self.layers[layer_name] = fc
            activation = nn.ReLU() if act == "relu" else ReLUN(act_clip_thres)
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            super().__setattr__(act_name, activation)
        layer_name = "fc"+str(len(self.hidden_list)+1)
        fc = BlockUSVLinear(self.hidden_list[-1], self.n_class, self.block_list[-1],
                            use_bias=False, S_trainable=self.S_trainable)
        super().__setattr__(layer_name, fc)
        self.layers[layer_name] = fc
        self.linears = nn.ModuleList(self.layers.values())

    def init_weights(self, initializer=nn.init.kaiming_normal_):
        for layer in self.layers:
            # print(self.layers[layer])
            self.layers[layer].init_weights()

    def init_weights_from_full_precision_model(self, model):
        for idx, layer in enumerate(model.layers):
            W = model.layers[layer].weight
            W_chunk = partition_chunks(
                W, bs=self.block_list[idx]).data.detach().cpu().numpy().astype(np.float64)
            U, S, V = np.linalg.svd(W_chunk, full_matrices=True)
            self.layers[layer].U.data.copy_(
                torch.from_numpy(U).to(torch.float32).to(self.device))
            self.layers[layer].S.data.copy_(
                torch.from_numpy(S).to(torch.float32).to(self.device).unsqueeze(-1))
            self.layers[layer].V.data.copy_(
                torch.from_numpy(V).to(torch.float32).to(self.device))

        print("[I] Initialize from full precision model")

    def add_weight_backup(self, cache_name):
        if(cache_name in self.weights_backup):
            print(f"[I] {cache_name} already exists, skip creation")
            return
        self.weights_backup[cache_name] = {layer: {} for layer in self.layers}
        print(f"[I] New cache {cache_name} created")

    def init_voltage_mask_and_backup(self):
        for layer in self.layers:
            out_block, in_block, block_size = self.layers[layer].out_block, self.layers[
                layer].in_block, self.layers[layer].block_size
            self.voltage_masks[layer] = {"U": np.zeros([out_block, in_block, block_size*(block_size-1)//2], dtype=np.bool),
                                         "V": np.zeros([out_block, in_block, block_size*(block_size-1)//2], dtype=np.bool)}
            self.voltage_backups[layer] = {"U": np.zeros([out_block, in_block, block_size*(block_size-1)//2], dtype=np.float64),
                                           "V": np.zeros([out_block, in_block, block_size*(block_size-1)//2], dtype=np.float64)}

    def getUSV(self):
        return {layer: {"U": self.layers[layer].U, "S": self.layers[layer].S, "V": self.layers[layer].V} for layer in self.layers}

    def getLipschitzLoss_DC(self):
        self.backupWeights()
        self.init_weights(initializer=nn.init.orthogonal_)
        self.lipschitz_loss_dc = self.getLipschitzLoss().data
        self.restoreWeights()

    def getLipschitzLoss(self, exclude_layers=[]):
        diff_list = []
        # print(exclude_layers)
        exclude_layers = set(exclude_layers)
        for layer in self.layers:
            if(layer in exclude_layers):
                continue
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            W = merge_chunks(self.layers[layer].buildWeight_from_USV(U, S, V))
            W_sq = torch.mm(W.t(), W)
            diff = W_sq - torch.eye(n=W_sq.size(0),
                                    dtype=W_sq.dtype, device=W_sq.device)
            # diff = self.layers[layer].S - 1
            diff_list.append(diff.contiguous().view(-1))
            # diff_sq_list.append(diff.abs().contiguous().view(-1))
        diff = torch.cat(diff_list, dim=0)
        diff_sq = diff * diff
        # loss = diff_sq.mean() - self.lipschitz_loss_dc
        loss = diff_sq.mean()

        return loss

    def getSigmaRegLoss(self):
        if(self.S_trainable):
            s_list = []
            for layer in self.layers:
                s_list.append(self.layers[layer].S.contiguous().view(-1))
            s_list = torch.cat(s_list, dim=0)
            loss = torch.mean(s_list * s_list)

    def getCrossLayerLipschitzLoss_DC(self):
        self.backupWeights()
        self.init_weights(initializer=nn.init.orthogonal_)
        self.cross_layer_lipschitz_loss_dc = self.getCrossLayerLipschitzLoss().data
        self.restoreWeights()

    def getCrossLayerLipschitzLoss(self):
        W_prod = None
        for layer in self.layers:
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            W = self.layers[layer].buildWeight_from_USV(U, S, V)
            W_prod = W if W_prod is None else torch.matmul(W, W_prod)
        W_prod_sq = torch.matmul(W_prod.t(), W_prod)
        diff = W_prod_sq - \
            torch.eye(n=W_prod_sq.size(0), dtype=W_prod_sq.dtype,
                      device=W_prod_sq.device)
        diff_sq = diff * diff
        loss = diff_sq.mean() - self.cross_layer_lipschitz_loss_dc
        return loss

    def getQuantizationLoss(self, quantize_voltage_percentile=1, hessian_dict=None):
        diff_list = []
        for layer in self.layers:
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            W = self.layers[layer].buildWeight_from_USV(U, S, V)
            U_q = quantize_voltage_of_unitary_cpu(
                U,
                v_bit=self.v_bit,
                v_pi=self.v_pi,
                v_max=self.v_max,
                voltage_mask=self.voltage_masks[layer]["U"],
                voltage_backup=self.voltage_backups[layer]["U"],
                quantize_voltage_percentile=quantize_voltage_percentile,
                output_device=self.device
            )
            V_q = quantize_voltage_of_unitary_cpu(
                V,
                v_bit=self.v_bit,
                v_pi=self.v_pi,
                v_max=self.v_max,
                voltage_mask=self.voltage_masks[layer]["V"],
                voltage_backup=self.voltage_backups[layer]["V"],
                quantize_voltage_percentile=quantize_voltage_percentile,
                output_device=self.device
            )
            W_q = self.layers[layer].buildWeight_from_USV(
                U_q.data, S.data, V_q.data).data

            diff = W - W_q.data
            diff_list.append(diff.contiguous().view(-1))
        diff = torch.cat(diff_list, dim=0)
        loss = (diff * diff).mean()

        return loss

    def backupWeights(self, cache_name="val_bk"):
        if(cache_name not in self.weights_backup):
            self.add_weight_backup(cache_name=cache_name)
        for layer in self.layers:
            self.weights_backup[cache_name][layer]["U"] = torch.clone(
                self.layers[layer].U.data)
            self.weights_backup[cache_name][layer]["S"] = torch.clone(
                self.layers[layer].S.data)
            self.weights_backup[cache_name][layer]["V"] = torch.clone(
                self.layers[layer].V.data)
        # print(f"[I] Weights {cache_name} backuped")

    def restoreWeights(self, cache_name='val_bk'):
        if(cache_name not in self.weights_backup):
            print(
                f"[W] No cache named {cache_name} to restore, skip restoring")
            return False
        for layer in self.layers:
            # assert layer in self.weights_backup[cache_name] and isinstance(self.weights_backup[cache_name][layer]["U"], torch.Tensor) and isinstance(self.weights_backup[cache_name][layer]["S"], torch.Tensor) and isinstance(self.weights_backup[cache_name][layer]["V"], torch.Tensor), "[E] Weight restore failed. Please call backupWeights before restoreWeights"
            if("U" in self.weights_backup[cache_name][layer]):
                self.layers[layer].U.data.copy_(
                    self.weights_backup[cache_name][layer]["U"])
            if("S" in self.weights_backup[cache_name][layer]):
                self.layers[layer].S.data.copy_(
                    self.weights_backup[cache_name][layer]["S"])
            if("V" in self.weights_backup[cache_name][layer]):
                self.layers[layer].V.data.copy_(
                    self.weights_backup[cache_name][layer]["V"])
        # print(f"[I] Weights {cache_name} restored from backup")
        return True

    def applyVoltageQuantization_kernel(self, U, V, voltage_mask_U=None, voltage_backup_U=None, voltage_mask_V=None, voltage_backup_V=None, quantize_voltage_percentile=1, strict_mask=True):
        U_q = quantize_voltage_of_unitary_cpu(
            U,
            v_bit=self.v_bit,
            v_pi=self.v_pi,
            v_max=self.v_max,
            voltage_mask=voltage_mask_U,
            voltage_backup=voltage_backup_U,
            quantize_voltage_percentile=quantize_voltage_percentile,
            strict_mask=strict_mask,
            output_device=self.device
        )
        V_q = quantize_voltage_of_unitary_cpu(
            V,
            v_bit=self.v_bit,
            v_pi=self.v_pi,
            v_max=self.v_max,
            voltage_mask=voltage_mask_V,
            voltage_backup=voltage_backup_V,
            quantize_voltage_percentile=quantize_voltage_percentile,
            strict_mask=strict_mask,
            output_device=self.device
        )
        U.data.copy_(U_q)
        V.data.copy_(V_q)

    def applyVoltageQuantization(self, quantize_voltage_percentile=1, with_mask=False, strict_mask=True):
        assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0,1]"
        for layer in self.layers:
            if(with_mask == True):
                self.applyVoltageQuantization_kernel(
                    self.layers[layer].U,
                    self.layers[layer].V,
                    self.voltage_masks[layer]["U"],
                    self.voltage_backups[layer]["U"],
                    self.voltage_masks[layer]["V"],
                    self.voltage_backups[layer]["V"],
                    quantize_voltage_percentile=quantize_voltage_percentile,
                    strict_mask=strict_mask
                )
            else:
                self.applyVoltageQuantization_kernel(
                    self.layers[layer].U,
                    self.layers[layer].V,
                    None, None, None, None,
                    quantize_voltage_percentile=quantize_voltage_percentile,
                    strict_mask=strict_mask
                )
        print("[I] Quantization applied")

    def applyVoltageQuantization_MT(self, quantize_voltage_percentile=1, with_mask=False, n_thread=8, strict_mask=True):
        assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0,1]"
        tasks = []
        for layer in self.layers:
            if(with_mask == True):
                tasks.append(
                    (self.layers[layer].U, self.voltage_masks[layer]["U"], self.voltage_backups[layer]["U"]))
                tasks.append(
                    (self.layers[layer].V, self.voltage_masks[layer]["V"], self.voltage_backups[layer]["V"]))
            else:
                tasks.append((self.layers[layer].U, None, None))
                tasks.append((self.layers[layer].V, None, None))
        self.pool.map(lambda args: args[0].data.copy_(quantize_voltage_of_unitary_cpu(
            args[0],
            v_bit=self.v_bit,
            v_pi=self.v_pi,
            v_max=self.v_max,
            voltage_mask=args[1],
            voltage_backup=args[2],
            quantize_voltage_percentile=quantize_voltage_percentile,
            strict_mask=strict_mask,
            output_device=self.device)),
            tasks)

    def applyVoltageMask(self, gamma_noise_std=0):
        for layer in self.layers:
            maintain_quantized_voltage_of_unitary_cpu(
                self.layers[layer].U,
                self.v_pi,
                self.voltage_masks[layer]["U"],
                self.voltage_backups[layer]["U"],
                gamma_noise_std=gamma_noise_std,
                output_device=self.device
            )
            maintain_quantized_voltage_of_unitary_cpu(
                self.layers[layer].V,
                self.v_pi,
                self.voltage_masks[layer]["V"],
                self.voltage_backups[layer]["V"],
                gamma_noise_std=gamma_noise_std,
                output_device=self.device
            )

    def applyVoltageMask_MT(self, gamma_noise_std=0, ori_cache_name=None, noise_cache_name=None, decay_rate=0, learning_rate=0, clip_voltage=False, thres_cache_name="clip_thres_bk"):
        tasks = []
        for layer in self.layers:
            tasks.append((self.layers[layer].U,
                          self.voltage_masks[layer]["U"],
                          self.voltage_backups[layer]["U"],
                          self.weights_backup[ori_cache_name][layer],
                          self.weights_backup[noise_cache_name][layer],
                          "U",
                          self.weights_backup[thres_cache_name][layer]["U"][0] if clip_voltage else float(
                              "-inf"),
                          self.weights_backup[thres_cache_name][layer]["U"][1] if clip_voltage else float(
                              "inf")
                          ))
            tasks.append((self.layers[layer].V,
                          self.voltage_masks[layer]["V"],
                          self.voltage_backups[layer]["V"],
                          self.weights_backup[ori_cache_name][layer],
                          self.weights_backup[noise_cache_name][layer],
                          "V",
                          self.weights_backup[thres_cache_name][layer]["V"][0] if clip_voltage else float(
                              "-inf"),
                          self.weights_backup[thres_cache_name][layer]["V"][1] if clip_voltage else float(
                              "inf")
                          ))

        def func(U, voltage_masks, voltage_backup, ori_weight_backup_dict, noise_weight_backup_dict, weight_backup_key, lower_thres, upper_thres):
            U_recon, U_recon_n = maintain_quantized_voltage_of_unitary_cpu(
                U,
                self.v_pi,
                voltage_masks,
                voltage_backup,
                gamma_noise_std=gamma_noise_std,
                weight_decay_rate=decay_rate,
                learning_rate=learning_rate,
                clip_voltage=clip_voltage,
                lower_thres=lower_thres,
                upper_thres=upper_thres,
                output_device=self.device
            )
            ori_weight_backup_dict[weight_backup_key] = U_recon.data.clone()
            if(U_recon_n is not None):
                noise_weight_backup_dict[weight_backup_key] = U_recon_n.data.clone(
                )

        self.pool.map(lambda args: func(*args), tasks)

    def applyUnitaryProjection(self):
        for layer in self.layers:
            self.layers[layer].U.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(
                self.layers[layer].U.data.detach().cpu().numpy())).to(self.device))
            self.layers[layer].V.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(
                self.layers[layer].V.data.detach().cpu().numpy())).to(self.device))

    def applyUnitaryProjection_MT(self):
        tasks = []
        for layer in self.layers:
            tasks.append(self.layers[layer].U)
            tasks.append(self.layers[layer].V)

        self.pool.map(lambda U: U.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(U.data.detach().cpu().numpy())).to(self.device)),
                      tasks)

        # print("[I] Unitary projection done")

    def applyConditionalUpdate_MT(self, lambda3, gamma_noise_std=0, ori_cache_name=None, noise_cache_name=None, decay_rate=0, learning_rate=0, clip_voltage=False, thres_cache_name="clip_thres_bk"):
        tasks = []
        for layer in self.layers:
            tasks.append((self.layers[layer].U,
                          self.weights_backup[ori_cache_name][layer],
                          self.weights_backup[noise_cache_name][layer],
                          "U",
                          self.weights_backup[thres_cache_name][layer]["U"][0] if clip_voltage else float(
                              "-inf"),
                          self.weights_backup[thres_cache_name][layer]["U"][1] if clip_voltage else float("inf")))
            tasks.append((self.layers[layer].V,
                          self.weights_backup[ori_cache_name][layer],
                          self.weights_backup[noise_cache_name][layer],
                          "V",
                          self.weights_backup[thres_cache_name][layer]["V"][0] if clip_voltage else float(
                              "-inf"),
                          self.weights_backup[thres_cache_name][layer]["V"][1] if clip_voltage else float("inf")))

        def func(U, ori_weight_backup_dict, noise_weight_backup_dict, weight_backup_key, lower_thres, upper_thres):
            U_recon, U_recon_n = conditional_update_voltage_of_unitary_cpu(
                W=U,
                v_bit=self.v_bit,
                v_pi=self.v_pi,
                v_max=self.v_max,
                lambda3=lambda3,
                voltage_mask=None,
                voltage_backup=None,
                gamma_noise_std=gamma_noise_std,
                weight_decay_rate=decay_rate,
                learning_rate=learning_rate,
                clip_voltage=clip_voltage,
                lower_thres=lower_thres,
                upper_thres=upper_thres,
                output_device=self.device)
            ori_weight_backup_dict[weight_backup_key] = U_recon.data.clone()
            if(U_recon_n is not None):
                noise_weight_backup_dict[weight_backup_key] = U_recon_n.data.clone(
                )

        self.pool.map(lambda args: func(*args), tasks)

    def addGammaNoise(self, gamma_noise_std=0):
        for layer in self.layers:
            U, V = self.layers[layer].U, self.layers[layer].V
            U_n = add_gamma_noise_to_unitary_cpu(
                U, v_pi=self.v_pi, gamma_noise_std=gamma_noise_std, output_device=self.device)
            V_n = add_gamma_noise_to_unitary_cpu(
                V, v_pi=self.v_pi, gamma_noise_std=gamma_noise_std, output_device=self.device)
            U.data.copy_(U_n)
            V.data.copy_(V_n)

    def calcClipVoltageThreshold(self, lower_perc=0, upper_perc=1, cache_name="clip_thres_bk"):

        if(cache_name not in self.weights_backup):
            self.add_weight_backup(cache_name)

        gamma = np.pi / (self.v_pi**2)
        quantizer = voltage_quantize_fn_cpu(
            v_bit=self.v_bit, v_pi=self.v_pi, v_max=self.v_max)
        for layer in self.layers:
            U, V = self.layers[layer].U, self.layers[layer].V
            W_numpy = U.detach().cpu().numpy().copy().astype(np.float64)

            N = W_numpy.shape[0]
            batch_mode = len(W_numpy.shape) > 2
            if(batch_mode):
                decomposer = RealUnitaryDecomposerBatch()
                delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
            else:
                decomposer = RealUnitaryDecomposer()
                delta_list, phi_mat = decomposer.decompose(W_numpy)
            phi_list = upper_triangle_to_vector_cpu(phi_mat)
            v_list = phase_to_voltage_cpu(phi_list, gamma)

            lower_thres = quantizer(np.percentile(v_list, lower_perc * 100))
            upper_thres = quantizer(np.percentile(v_list, upper_perc * 100))

            self.weights_backup[cache_name][layer]["U"] = (
                lower_thres, upper_thres)

            W_numpy = V.detach().cpu().numpy().copy().astype(np.float64)
            N = W_numpy.shape[0]
            if(batch_mode):
                delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
            else:
                delta_list, phi_mat = decomposer.decompose(W_numpy)
            phi_list = upper_triangle_to_vector_cpu(phi_mat)
            v_list = phase_to_voltage_cpu(phi_list, gamma)

            lower_thres = quantizer(np.percentile(v_list, lower_perc * 100))
            upper_thres = quantizer(np.percentile(v_list, upper_perc * 100))
            self.weights_backup[cache_name][layer]["V"] = (
                lower_thres, upper_thres)

        print("[I] Voltage clipping thresholds are cached")
        print(self.weights_backup[cache_name])

    def initClipThresScheduler(self, step_beg, step_end, beg_cache="clip_thres_beg_bk", end_cache="clip_thres_end_bk"):
        thres_cache = "clip_thres_scheduler_bk"
        self.add_weight_backup(thres_cache)
        for layer in self.layers:

            lower_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["U"][0], self.weights_backup[end_cache][layer]["U"][0])
            upper_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["U"][1], self.weights_backup[end_cache][layer]["U"][1])
            self.weights_backup[thres_cache][layer]["U"] = (
                lower_scheduler, upper_scheduler)

            lower_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["V"][0], self.weights_backup[end_cache][layer]["V"][0])
            upper_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["V"][1], self.weights_backup[end_cache][layer]["V"][1])
            self.weights_backup[thres_cache][layer]["V"] = (
                lower_scheduler, upper_scheduler)

    def updateClipVoltageThreshold(self, epoch, thres_cache="clip_thres_bk"):
        scheduler_cache = "clip_thres_scheduler_bk"
        for layer in self.layers:
            lower_scheduler = self.weights_backup[scheduler_cache][layer]["U"][0]
            upper_scheduler = self.weights_backup[scheduler_cache][layer]["U"][1]
            self.weights_backup[thres_cache][layer]["U"] = (
                lower_scheduler(epoch), upper_scheduler(epoch))

            lower_scheduler = self.weights_backup[scheduler_cache][layer]["V"][0]
            upper_scheduler = self.weights_backup[scheduler_cache][layer]["V"][1]
            self.weights_backup[thres_cache][layer]["V"] = (
                lower_scheduler(epoch), upper_scheduler(epoch))

    def clipVoltages_MT(self, cache_name="clip_thres_bk"):
        tasks = []
        for layer in self.layers:
            U, V = self.layers[layer].U, self.layers[layer].V
            thres_U = self.weights_backup[cache_name][layer]["U"]
            thres_V = self.weights_backup[cache_name][layer]["V"]
            tasks.append([U, thres_U[0], thres_U[1]])
            tasks.append([V, thres_V[0], thres_V[1]])

        self.pool.map(lambda args: args[0].data.copy_(clip_voltage_of_unitary_cpu(
            args[0], v_pi=self.v_pi, lower_thres=args[1], upper_thres=args[2], output_device=self.device)), tasks)

        # print(f"[I] Voltages are pruned, ({lower_perc}, {upper_perc})")

    def forward(self, x):
        # x = x[..., ::4, ::4].contiguous()
        # x = F.avg_pool2d(x, kernel_size=4, stride=4, padding=0).contiguous()
        x = x.view(-1, self.n_feat)
        n_layer = len(self.layers)
        for idx, layer in enumerate(self.layers):
            fc = self.layers[layer]
            x = fc(x)
            if(idx + 1 < n_layer):
                act = self.acts[layer]
                x = act(x)

        out = F.log_softmax(x, dim=1)
        return out
        # x = self.fc1(x)
        # # x = x.clamp(0, 4)
        # x = self.act1(x)
        # x = self.fc2(x)

        # return F.log_softmax(x, dim=1)


class FCMLP_BlockUSV_Q_Dorefa(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, hidden_list=[16], block_list=[8, 10], S_trainable=True, v_bit=4, v_pi=4.36, v_max=10.8, clamp_small_phase_lead_percentile=1, n_thread=4, act="relu", act_thres=4, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.hidden_list = hidden_list
        self.block_list = block_list
        self.S_trainable = S_trainable
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.act = act
        self.act_thres = act_thres
        self.clamp_small_phase_lead_percentile = clamp_small_phase_lead_percentile
        self.device = device
        self.buildLayers(act=act, act_clip_thres=act_thres)
        # self.act1 = ReLUN(4)
        # self.act1 = nn.ReLU()
        # self.layers = {"fc1": self.fc1, "fc2": self.fc2}
        self.weights_backup = {}
        self.add_weight_backup(cache_name="val_bk")
        # self.weights_backup = {'val_bk': {"fc1":{}, "fc2":{}}}
        self.voltage_masks = {}
        self.voltage_backups = {}
        self.lipschitz_loss_dc = 0
        self.cross_layer_lipschitz_loss_dc = 0
        self.init_weights(initializer=nn.init.orthogonal_)
        self.init_voltage_mask_and_backup()
        # self.getLipschitzLoss_DC()
        # self.getCrossLayerLipschitzLoss_DC()
        self.pool = Pool(n_thread)
        self.ema = EMA(0.999)
        # self.decomposer = RealUniaryDecomposer()
        # self.phase_quantize_fn = phase_quantize_fn_cpu(p_bit=4)

    def buildLayers(self, act="relun", act_clip_thres=4):
        # assert act in {"relu", "relun"}, f"[E] Not supported activation function: {act}"
        # assert act_clip_thres > 0, f"[E] Threshold {act_clip_thres} is not supported by ReLUN()"
        self.layers = OrderedDict()
        self.acts = OrderedDict()
        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            act_name = "act" + str(idx+1)
            in_channel = self.n_feat if idx == 0 else self.hidden_list[idx-1]
            out_channel = hidden_dim
            fc = BlockUSVLinear(in_channel, out_channel, self.block_list[idx],
                                use_bias=False, S_trainable=self.S_trainable, device=self.device)
            self.layers[layer_name] = fc
            activation = nn.ReLU() if act == "relu" else ReLUN(act_clip_thres)
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            super().__setattr__(act_name, activation)
        layer_name = "fc"+str(len(self.hidden_list)+1)
        fc = BlockUSVLinear(self.hidden_list[-1], self.n_class, self.block_list[-1],
                            use_bias=False, S_trainable=self.S_trainable, device=self.device)
        super().__setattr__(layer_name, fc)
        self.layers[layer_name] = fc
        self.linears = nn.ModuleList(self.layers.values())

    def init_weights(self, initializer=nn.init.kaiming_normal_):
        for layer in self.layers:
            # print(self.layers[layer])
            self.layers[layer].init_weights()

    def init_weights_from_full_precision_model(self, model):
        for idx, layer in enumerate(model.layers):
            W = model.layers[layer].weight
            W_chunk = partition_chunks(
                W, bs=self.block_list[idx]).data.detach().cpu().numpy().astype(np.float64)
            U, S, V = np.linalg.svd(W_chunk, full_matrices=True)
            self.layers[layer].U.data.copy_(
                torch.from_numpy(U).to(torch.float32).to(self.device))
            self.layers[layer].S.data.copy_(
                torch.from_numpy(S).to(torch.float32).to(self.device).unsqueeze(-1))
            self.layers[layer].V.data.copy_(
                torch.from_numpy(V).to(torch.float32).to(self.device))

        print("[I] Initialize from full precision model")

    def add_weight_backup(self, cache_name):
        if(cache_name in self.weights_backup):
            print(f"[I] {cache_name} already exists, skip creation")
            return
        self.weights_backup[cache_name] = {layer: {} for layer in self.layers}
        print(f"[I] New cache {cache_name} created")

    def init_voltage_mask_and_backup(self):
        for layer in self.layers:
            out_block, in_block, block_size = self.layers[layer].out_block, self.layers[
                layer].in_block, self.layers[layer].block_size
            self.voltage_masks[layer] = {"U": np.zeros([out_block, in_block, block_size*(block_size-1)//2], dtype=np.bool),
                                         "V": np.zeros([out_block, in_block, block_size*(block_size-1)//2], dtype=np.bool)}
            self.voltage_backups[layer] = {"U": np.zeros([out_block, in_block, block_size*(block_size-1)//2], dtype=np.float64),
                                           "V": np.zeros([out_block, in_block, block_size*(block_size-1)//2], dtype=np.float64)}

    def getUSV(self):
        return {layer: {"U": self.layers[layer].U, "S": self.layers[layer].S, "V": self.layers[layer].V} for layer in self.layers}

    def getLipschitzLoss_DC(self):
        self.backupWeights()
        self.init_weights(initializer=nn.init.orthogonal_)
        self.lipschitz_loss_dc = self.getLipschitzLoss().data
        self.restoreWeights()

    def getLipschitzLoss(self, quant_uv_cache_name="quant_uv_bk"):
        diff_list = []
        for layer in self.layers:
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            # U = self.weights_backup[quant_uv_cache_name][layer]["U"]
            # S = self.layers[layer].S
            # V = self.weights_backup[quant_uv_cache_name][layer]["V"]
            W = merge_chunks(self.layers[layer].buildWeight_from_USV(U, S, V))
            W_sq = torch.mm(W.t(), W)
            diff = W_sq - torch.eye(n=W_sq.size(0),
                                    dtype=W_sq.dtype, device=W_sq.device)
            diff_list.append(diff.contiguous().view(-1))
            # diff_sq_list.append(diff.abs().contiguous().view(-1))
        diff = torch.cat(diff_list, dim=0)
        diff_sq = diff * diff
        loss = diff_sq.sum()

        return loss

    def getSigmaL1RegLoss(self):
        if(self.S_trainable):
            s_list = []
            for layer in self.layers:
                s_list.append(self.layers[layer].S.contiguous().view(-1))
            s_list = torch.cat(s_list, dim=0)
            loss = s_list.abs().sum()
        return loss

    def getGroupLassoLoss(self):
        loss = None
        for layer in self.layers:
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            W = self.layers[layer].buildWeight_from_USV(U, S, V)
            M, N = W.size(0), W.size(1)
            W = W.contiguous().view(M, N, -1)
            P = W.size(2)

            lasso = torch.norm(W, p=2, dim=2).sum() / P
            loss = lasso if loss is None else loss + lasso
        return loss

    def getProtectiveGroupLassoLoss(self, quant_uv_cache_name="quant_uv_bk", ori_uv_cache_name="ori_uv_bk", prune_mask_cache_name="prune_mask"):
        quant_uv_cache = self.weights_backup[quant_uv_cache_name]
        ori_uv_cache = self.weights_backup[ori_uv_cache_name]
        prune_mask_cache = self.weights_backup[prune_mask_cache_name]
        loss = None
        for layer in self.layers:
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            W = self.layers[layer].buildWeight_from_USV(U, S, V)
            lasso = torch.norm(W, p=2, dim=[-1, -2])

            U_q = ori_uv_cache[layer]["U"].data
            V_q = ori_uv_cache[layer]["V"].data
            U_qn = quant_uv_cache[layer]["U"].data
            V_qn = quant_uv_cache[layer]["V"].data
            W_q = self.layers[layer].buildWeight_from_USV(U_q, S.data, V_q)
            W_qn = self.layers[layer].buildWeight_from_USV(U_qn, S.data, V_qn)
            distance = torch.norm(W_q - W_qn, p=2, dim=[-1, -2])
            coeff = distance / distance.max()
            mask = prune_mask_cache[layer]["mask"]
            coeff = self.ema(layer+"_coeff", coeff, mask)
            lasso = (coeff.data * lasso).sum() / (W.size(-2) * W.size(-1))
            loss = lasso if loss is None else loss + lasso
        return loss

    def initPruneMask(self, mask_cache_name="prune_mask"):
        cache = self.weights_backup[mask_cache_name]
        for layer in self.layers:
            U = self.layers[layer].U
            cache[layer]["mask"] = torch.zeros(U.size(0), U.size(1), dtype=torch.bool, device=self.device)

    def updatePruneMask(self, perc, mask_cache_name="prune_mask"):
        cache = self.weights_backup[mask_cache_name]
        for layer in self.layers:
            coeff = self.ema.shadow[layer+"_coeff"]
            thres = np.percentile(coeff.detach().cpu().numpy(), (1-perc)*100)
            mask = coeff > thres
            cache[layer]["mask"] = mask
            coeff.data[mask] = 1 # force the masked coefficients to maximum value 1, so they will always be in the pruned range

    def pruneWeights(self, mask_cache_name="prune_mask"):
        cache = self.weights_backup[mask_cache_name]
        for layer in self.layers:
            mask = cache[layer]["mask"]
            U, S, V = self.layers[layer].U, self.layers[layer].S, self.layers[layer].V
            U.data[mask, :, :], S.data[mask, :], V.data[mask, :, :] = 0, 0, 0

    def getCrossLayerLipschitzLoss_DC(self):
        self.backupWeights()
        self.init_weights(initializer=nn.init.orthogonal_)
        self.cross_layer_lipschitz_loss_dc = self.getCrossLayerLipschitzLoss().data
        self.restoreWeights()

    def getCrossLayerLipschitzLoss(self):
        W_prod = None
        for layer in self.layers:
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            W = self.layers[layer].buildWeight_from_USV(U, S, V)
            W_prod = W if W_prod is None else torch.matmul(W, W_prod)
        W_prod_sq = torch.matmul(W_prod.t(), W_prod)
        diff = W_prod_sq - \
            torch.eye(n=W_prod_sq.size(0), dtype=W_prod_sq.dtype,
                      device=W_prod_sq.device)
        diff_sq = diff * diff
        loss = diff_sq.mean() - self.cross_layer_lipschitz_loss_dc
        return loss

    def getWeightPNormRegLoss(self, p=2):
        norm_list = []
        for layer in self.layers:
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            W = merge_chunks(self.layers[layer].buildWeight_from_USV(U, S, V))
            norm_list.append(W.contiguous().view(-1))
        norm = torch.cat(norm_list, dim=0)
        loss = torch.pow(norm, 2).sum()
        return loss

    def getQuantizationLoss(self, cache_name="quant_uv_bk"):
        diff_list = []
        cache = self.weights_backup[cache_name]
        for layer in self.layers:
            U = self.layers[layer].U
            S = self.layers[layer].S
            V = self.layers[layer].V
            W = self.layers[layer].buildWeight_from_USV(U, S, V)
            U_q = cache[layer]["U"]
            V_q = cache[layer]["V"]
            W_q = self.layers[layer].buildWeight_from_USV(
                U_q.data, S.data, V_q.data).data

            diff = W - W_q.data
            diff_list.append(diff.contiguous().view(-1))
        diff = torch.cat(diff_list, dim=0)
        loss = (diff * diff).mean()

        return loss

    def backupWeights(self, cache_name="val_bk"):
        if(cache_name not in self.weights_backup):
            self.add_weight_backup(cache_name=cache_name)
        for layer in self.layers:
            self.weights_backup[cache_name][layer]["U"] = torch.clone(
                self.layers[layer].U.data)
            self.weights_backup[cache_name][layer]["S"] = torch.clone(
                self.layers[layer].S.data)
            self.weights_backup[cache_name][layer]["V"] = torch.clone(
                self.layers[layer].V.data)
        # print(f"[I] Weights {cache_name} backuped")

    def restoreWeights(self, cache_name='val_bk'):
        if(cache_name not in self.weights_backup):
            print(
                f"[W] No cache named {cache_name} to restore, skip restoring")
            return False
        for layer in self.layers:
            # assert layer in self.weights_backup[cache_name] and isinstance(self.weights_backup[cache_name][layer]["U"], torch.Tensor) and isinstance(self.weights_backup[cache_name][layer]["S"], torch.Tensor) and isinstance(self.weights_backup[cache_name][layer]["V"], torch.Tensor), "[E] Weight restore failed. Please call backupWeights before restoreWeights"
            if("U" in self.weights_backup[cache_name][layer]):
                self.layers[layer].U.data.copy_(
                    self.weights_backup[cache_name][layer]["U"])
            if("S" in self.weights_backup[cache_name][layer]):
                self.layers[layer].S.data.copy_(
                    self.weights_backup[cache_name][layer]["S"])
            if("V" in self.weights_backup[cache_name][layer]):
                self.layers[layer].V.data.copy_(
                    self.weights_backup[cache_name][layer]["V"])
        # print(f"[I] Weights {cache_name} restored from backup")
        return True

    def applyVoltageQuantization_kernel(self, U, V, voltage_mask_U=None, voltage_backup_U=None, voltage_mask_V=None, voltage_backup_V=None, quantize_voltage_percentile=1, strict_mask=True):
        U_q = quantize_voltage_of_unitary_cpu(
            U,
            v_bit=self.v_bit,
            v_pi=self.v_pi,
            v_max=self.v_max,
            voltage_mask=voltage_mask_U,
            voltage_backup=voltage_backup_U,
            quantize_voltage_percentile=quantize_voltage_percentile,
            strict_mask=strict_mask,
            output_device=self.device
        )
        V_q = quantize_voltage_of_unitary_cpu(
            V,
            v_bit=self.v_bit,
            v_pi=self.v_pi,
            v_max=self.v_max,
            voltage_mask=voltage_mask_V,
            voltage_backup=voltage_backup_V,
            quantize_voltage_percentile=quantize_voltage_percentile,
            strict_mask=strict_mask,
            output_device=self.device
        )
        U.data.copy_(U_q)
        V.data.copy_(V_q)

    def applyVoltageQuantization(self, quantize_voltage_percentile=1, with_mask=False, strict_mask=True):
        assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0,1]"
        for layer in self.layers:
            if(with_mask == True):
                self.applyVoltageQuantization_kernel(
                    self.layers[layer].U,
                    self.layers[layer].V,
                    self.voltage_masks[layer]["U"],
                    self.voltage_backups[layer]["U"],
                    self.voltage_masks[layer]["V"],
                    self.voltage_backups[layer]["V"],
                    quantize_voltage_percentile=quantize_voltage_percentile,
                    strict_mask=strict_mask
                )
            else:
                self.applyVoltageQuantization_kernel(
                    self.layers[layer].U,
                    self.layers[layer].V,
                    None, None, None, None,
                    quantize_voltage_percentile=quantize_voltage_percentile,
                    strict_mask=strict_mask
                )
        print("[I] Quantization applied")

    def applyVoltageQuantization_MT(self, quantize_voltage_percentile=1, with_mask=False, n_thread=8, strict_mask=True):
        assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0,1]"
        tasks = []
        for layer in self.layers:
            if(with_mask == True):
                tasks.append(
                    (self.layers[layer].U, self.voltage_masks[layer]["U"], self.voltage_backups[layer]["U"]))
                tasks.append(
                    (self.layers[layer].V, self.voltage_masks[layer]["V"], self.voltage_backups[layer]["V"]))
            else:
                tasks.append((self.layers[layer].U, None, None))
                tasks.append((self.layers[layer].V, None, None))
        self.pool.map(lambda args: args[0].data.copy_(quantize_voltage_of_unitary_cpu(
            args[0],
            v_bit=self.v_bit,
            v_pi=self.v_pi,
            v_max=self.v_max,
            voltage_mask=args[1],
            voltage_backup=args[2],
            quantize_voltage_percentile=quantize_voltage_percentile,
            strict_mask=strict_mask,
            output_device=self.device)),
            tasks)

    def applyVoltageMask(self, gamma_noise_std=0):
        for layer in self.layers:
            maintain_quantized_voltage_of_unitary_cpu(
                self.layers[layer].U,
                self.v_pi,
                self.voltage_masks[layer]["U"],
                self.voltage_backups[layer]["U"],
                gamma_noise_std=gamma_noise_std,
                output_device=self.device
            )
            maintain_quantized_voltage_of_unitary_cpu(
                self.layers[layer].V,
                self.v_pi,
                self.voltage_masks[layer]["V"],
                self.voltage_backups[layer]["V"],
                gamma_noise_std=gamma_noise_std,
                output_device=self.device
            )

    def applyVoltageMask_MT(self, gamma_noise_std=0, ori_cache_name=None, noise_cache_name=None, decay_rate=0, learning_rate=0, clip_voltage=False, thres_cache_name="clip_thres_bk"):
        tasks = []
        for layer in self.layers:
            tasks.append((self.layers[layer].U,
                          self.voltage_masks[layer]["U"],
                          self.voltage_backups[layer]["U"],
                          self.weights_backup[ori_cache_name][layer],
                          self.weights_backup[noise_cache_name][layer],
                          "U",
                          self.weights_backup[thres_cache_name][layer]["U"][0] if clip_voltage else float(
                              "-inf"),
                          self.weights_backup[thres_cache_name][layer]["U"][1] if clip_voltage else float(
                              "inf")
                          ))
            tasks.append((self.layers[layer].V,
                          self.voltage_masks[layer]["V"],
                          self.voltage_backups[layer]["V"],
                          self.weights_backup[ori_cache_name][layer],
                          self.weights_backup[noise_cache_name][layer],
                          "V",
                          self.weights_backup[thres_cache_name][layer]["V"][0] if clip_voltage else float(
                              "-inf"),
                          self.weights_backup[thres_cache_name][layer]["V"][1] if clip_voltage else float(
                              "inf")
                          ))

        def func(U, voltage_masks, voltage_backup, ori_weight_backup_dict, noise_weight_backup_dict, weight_backup_key, lower_thres, upper_thres):
            U_recon, U_recon_n = maintain_quantized_voltage_of_unitary_cpu(
                U,
                self.v_pi,
                voltage_masks,
                voltage_backup,
                gamma_noise_std=gamma_noise_std,
                weight_decay_rate=decay_rate,
                learning_rate=learning_rate,
                clip_voltage=clip_voltage,
                lower_thres=lower_thres,
                upper_thres=upper_thres,
                output_device=self.device
            )
            ori_weight_backup_dict[weight_backup_key] = U_recon.data.clone()
            if(U_recon_n is not None):
                noise_weight_backup_dict[weight_backup_key] = U_recon_n.data.clone(
                )

        self.pool.map(lambda args: func(*args), tasks)

    def applyUnitaryProjection(self):
        for layer in self.layers:
            self.layers[layer].U.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(
                self.layers[layer].U.data.detach().cpu().numpy())).to(self.device))
            self.layers[layer].V.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(
                self.layers[layer].V.data.detach().cpu().numpy())).to(self.device))

    def applyUnitaryProjection_MT(self):
        tasks = []
        for layer in self.layers:
            tasks.append(self.layers[layer].U)
            tasks.append(self.layers[layer].V)

        self.pool.map(lambda U: U.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(U.data.detach().cpu().numpy())).to(self.device)),
                      tasks)

        # print("[I] Unitary projection done")

    def applyConditionalUpdate_MT(self, lambda3, gamma_noise_std=0, ori_cache_name=None, noise_cache_name=None, decay_rate=0, learning_rate=0, clip_voltage=False, thres_cache_name="clip_thres_bk"):
        tasks = []
        for layer in self.layers:
            tasks.append((self.layers[layer].U,
                          self.weights_backup[ori_cache_name][layer],
                          self.weights_backup[noise_cache_name][layer],
                          "U",
                          self.weights_backup[thres_cache_name][layer]["U"][0] if clip_voltage else float(
                              "-inf"),
                          self.weights_backup[thres_cache_name][layer]["U"][1] if clip_voltage else float("inf")))
            tasks.append((self.layers[layer].V,
                          self.weights_backup[ori_cache_name][layer],
                          self.weights_backup[noise_cache_name][layer],
                          "V",
                          self.weights_backup[thres_cache_name][layer]["V"][0] if clip_voltage else float(
                              "-inf"),
                          self.weights_backup[thres_cache_name][layer]["V"][1] if clip_voltage else float("inf")))

        def func(U, ori_weight_backup_dict, noise_weight_backup_dict, weight_backup_key, lower_thres, upper_thres):
            U_recon, U_recon_n = conditional_update_voltage_of_unitary_cpu(
                W=U,
                v_bit=self.v_bit,
                v_pi=self.v_pi,
                v_max=self.v_max,
                lambda3=lambda3,
                voltage_mask=None,
                voltage_backup=None,
                gamma_noise_std=gamma_noise_std,
                weight_decay_rate=decay_rate,
                learning_rate=learning_rate,
                clip_voltage=clip_voltage,
                lower_thres=lower_thres,
                upper_thres=upper_thres,
                output_device=self.device)
            ori_weight_backup_dict[weight_backup_key] = U_recon.data.clone()
            if(U_recon_n is not None):
                noise_weight_backup_dict[weight_backup_key] = U_recon_n.data.clone(
                )

        self.pool.map(lambda args: func(*args), tasks)

    def addGammaNoise(self, gamma_noise_std=0):
        for layer in self.layers:
            U, V = self.layers[layer].U, self.layers[layer].V
            U_n = add_gamma_noise_to_unitary_cpu(
                U, v_pi=self.v_pi, gamma_noise_std=gamma_noise_std, output_device=self.device)
            V_n = add_gamma_noise_to_unitary_cpu(
                V, v_pi=self.v_pi, gamma_noise_std=gamma_noise_std, output_device=self.device)
            U.data.copy_(U_n)
            V.data.copy_(V_n)

    def calcClipVoltageThreshold(self, lower_perc=0, upper_perc=1, cache_name="clip_thres_bk"):

        if(cache_name not in self.weights_backup):
            self.add_weight_backup(cache_name)

        gamma = np.pi / (self.v_pi**2)
        quantizer = voltage_quantize_fn_cpu(
            v_bit=self.v_bit, v_pi=self.v_pi, v_max=self.v_max)
        for layer in self.layers:
            U, V = self.layers[layer].U, self.layers[layer].V
            W_numpy = U.detach().cpu().numpy().copy().astype(np.float64)

            N = W_numpy.shape[0]
            batch_mode = len(W_numpy.shape) > 2
            if(batch_mode):
                decomposer = RealUnitaryDecomposerBatch()
                delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
            else:
                decomposer = RealUnitaryDecomposer()
                delta_list, phi_mat = decomposer.decompose(W_numpy)
            phi_list = upper_triangle_to_vector_cpu(phi_mat)
            v_list = phase_to_voltage_cpu(phi_list, gamma)

            lower_thres = quantizer(np.percentile(v_list, lower_perc * 100))
            upper_thres = quantizer(np.percentile(v_list, upper_perc * 100))

            self.weights_backup[cache_name][layer]["U"] = (
                lower_thres, upper_thres)

            W_numpy = V.detach().cpu().numpy().copy().astype(np.float64)
            N = W_numpy.shape[0]
            if(batch_mode):
                delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
            else:
                delta_list, phi_mat = decomposer.decompose(W_numpy)
            phi_list = upper_triangle_to_vector_cpu(phi_mat)
            v_list = phase_to_voltage_cpu(phi_list, gamma)

            lower_thres = quantizer(np.percentile(v_list, lower_perc * 100))
            upper_thres = quantizer(np.percentile(v_list, upper_perc * 100))
            self.weights_backup[cache_name][layer]["V"] = (
                lower_thres, upper_thres)

        print("[I] Voltage clipping thresholds are cached")
        print(self.weights_backup[cache_name])

    def initClipThresScheduler(self, step_beg, step_end, beg_cache="clip_thres_beg_bk", end_cache="clip_thres_end_bk"):
        thres_cache = "clip_thres_scheduler_bk"
        self.add_weight_backup(thres_cache)
        for layer in self.layers:

            lower_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["U"][0], self.weights_backup[end_cache][layer]["U"][0])
            upper_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["U"][1], self.weights_backup[end_cache][layer]["U"][1])
            self.weights_backup[thres_cache][layer]["U"] = (
                lower_scheduler, upper_scheduler)

            lower_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["V"][0], self.weights_backup[end_cache][layer]["V"][0])
            upper_scheduler = ThresholdScheduler_tf(
                step_beg, step_end, self.weights_backup[beg_cache][layer]["V"][1], self.weights_backup[end_cache][layer]["V"][1])
            self.weights_backup[thres_cache][layer]["V"] = (
                lower_scheduler, upper_scheduler)

    def updateClipVoltageThreshold(self, epoch, thres_cache="clip_thres_bk"):
        scheduler_cache = "clip_thres_scheduler_bk"
        for layer in self.layers:
            lower_scheduler = self.weights_backup[scheduler_cache][layer]["U"][0]
            upper_scheduler = self.weights_backup[scheduler_cache][layer]["U"][1]
            self.weights_backup[thres_cache][layer]["U"] = (
                lower_scheduler(epoch), upper_scheduler(epoch))

            lower_scheduler = self.weights_backup[scheduler_cache][layer]["V"][0]
            upper_scheduler = self.weights_backup[scheduler_cache][layer]["V"][1]
            self.weights_backup[thres_cache][layer]["V"] = (
                lower_scheduler(epoch), upper_scheduler(epoch))

    def clipVoltages_MT(self, cache_name="clip_thres_bk"):
        tasks = []
        for layer in self.layers:
            U, V = self.layers[layer].U, self.layers[layer].V
            thres_U = self.weights_backup[cache_name][layer]["U"]
            thres_V = self.weights_backup[cache_name][layer]["V"]
            tasks.append([U, thres_U[0], thres_U[1]])
            tasks.append([V, thres_V[0], thres_V[1]])

        self.pool.map(lambda args: args[0].data.copy_(clip_voltage_of_unitary_cpu(
            args[0], v_pi=self.v_pi, lower_thres=args[1], upper_thres=args[2], output_device=self.device)), tasks)

        # print(f"[I] Voltages are pruned, ({lower_perc}, {upper_perc})")

    def conditional_quantize_with_gamma_noise_layer(self, layer, lambda3, gamma_noise_std=0, decay_rate=0, learning_rate=0, clip_voltage=False, thres_cache_name="clip_thres_bk", prune_mask_cache_name="prune_mask"):
        if(clip_voltage):
            thres_cache = self.weights_backup[thres_cache_name]
        U, V = self.layers[layer].U, self.layers[layer].V
        mask = self.weights_backup[prune_mask_cache_name][layer]["mask"]


        U_q, U_n = ConditionalQuantizationWithGammaNoiseOfUnitary.apply(
            U,
            self.v_bit,
            self.v_pi,
            self.v_max,
            lambda3,
            gamma_noise_std,
            decay_rate,
            learning_rate,
            clip_voltage,
            thres_cache[layer]["U"][0] if clip_voltage else float("-inf"),
            thres_cache[layer]["U"][1] if clip_voltage else float("inf"),
            mask,
            self.device)
        V_q, V_n = ConditionalQuantizationWithGammaNoiseOfUnitary.apply(
            V,
            self.v_bit,
            self.v_pi,
            self.v_max,
            lambda3,
            gamma_noise_std,
            decay_rate,
            learning_rate,
            clip_voltage,
            thres_cache[layer]["V"][0] if clip_voltage else float("-inf"),
            thres_cache[layer]["V"][1] if clip_voltage else float("inf"),
            mask,
            self.device)

        return U_q, V_q, U_n, V_n

    def forward(self, x, lambda3=0, gamma_noise_std=0, decay_rate=0, learning_rate=0, clip_voltage=False, thres_cache_name="clip_thres_bk", conditional_quant=True, quant_uv_cache_name="quant_uv_bk", ori_uv_cache_name="ori_uv_bk", prune_mask_cache_name="prune_mask"):
        quant_uv_cache = self.weights_backup[quant_uv_cache_name]
        ori_uv_cache = self.weights_backup[ori_uv_cache_name]

        x = x.view(-1, self.n_feat)

        n_layer = len(self.layers)
        for idx, layer in enumerate(self.layers):
            fc = self.layers[layer]
            if(conditional_quant):
                U_q, V_q, U_n, V_n = self.conditional_quantize_with_gamma_noise_layer(
                    layer, lambda3, gamma_noise_std=gamma_noise_std, decay_rate=decay_rate, learning_rate=learning_rate, clip_voltage=clip_voltage, thres_cache_name=thres_cache_name, prune_mask_cache_name=prune_mask_cache_name)


                quant_uv_cache[layer]["U"] = U_n
                quant_uv_cache[layer]["V"] = V_n
                ori_uv_cache[layer]["U"] = U_q
                ori_uv_cache[layer]["V"] = V_q

                x = fc(x, U_n, V_n)
            else:
                x = fc(x)
            if(idx + 1 < n_layer):
                act = self.acts[layer]
                x = act(x)

        out = F.log_softmax(x, dim=1)
        return out


class FCMLP_3Layer(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.device = device
        self.fc1 = nn.Linear(n_feat, 128, bias=False)
        self.fc2 = nn.Linear(128, 128, bias=False)
        self.fc3 = nn.Linear(128, 10, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        # x = x[...,::2,::2].contiguous()
        x = x.view(-1, self.n_feat)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=1)


class FCMLP_v2(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.device = device
        self.fc1 = nn.Linear(n_feat, 200, bias=False)
        self.fc2 = nn.Linear(200, 200, bias=False)
        self.fc3 = nn.Linear(200, 10, bias=False)
        self.layers = {"fc1": self.fc1, "fc2": self.fc2, "fc3": self.fc3}
        self.weights_backup = {}
        self.init_weights()
        self.decomposer = RealUniaryDecomposerPyTorch(timer=False)

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

    def getLassoLoss(self, sparsity=None):
        lasso = {
            layer: self.layers[layer].weight.data.abs() for layer in self.layers}
        lasso = torch.cat([i.contiguous().view(-1)
                           for i in lasso.values()], 0)

        lasso_loss = torch.mean(lasso)
        if(sparsity is not None):
            lasso_loss = lasso_loss  # / (1 - sparsity)

        return lasso_loss

    def getDropMask(self, drop_thres=0.01):
        lasso = {
            layer: self.layers[layer].weight.data.abs() for layer in self.layers}
        assert drop_thres >= 0, "[E] Drop threshold must be non-negative a real number"
        self.drop_masks = {layer: lasso >=
                           drop_thres for layer, lasso in lasso.items()}

        return self.drop_masks

    def applyDropMask(self, drop_masks=None):
        if(not drop_masks):
            return
        for layer, mask in drop_masks.items():
            self.layers[layer].weight.data = self.layers[layer].weight.data * \
                mask.to(torch.float32)

    def backupWeights(self):
        self.weights_backup = {}
        for layer in self.layers:
            self.weights_backup[layer] = torch.clone(
                self.layers[layer].weight.data)
        print(f"[I] Weights backuped")

    def prunePhase(self, epsilon=1e-3):
        n_prune = 0
        for layer in self.layers:
            W = self.layers[layer].weight.data
            M, N = W.size(0), W.size(1)
            U, Sigma, V = np.linalg.svd(W.cpu().numpy(), full_matrices=True)
            U = torch.from_numpy(U).to(self.device)
            Sigma = np.diag(Sigma)
            if(M > N):
                Sigma = np.concatenate([Sigma, np.zeros([M - N, N])], axis=0)
            elif(M < N):
                Sigma = np.concatenate([Sigma, np.zeros([M, N - M])], axis=1)
            Sigma = torch.from_numpy(Sigma).to(self.device).float()
            V = torch.from_numpy(V).to(self.device)
            delta_list_U, phi_mat_U = self.decomposer.decompose(U)
            delta_list_V, phi_mat_V = self.decomposer.decompose(V)
            # with fullprint(threshold=None, linewidth=400, precision=2):
            #     print(phi_mat_V.cpu().numpy().tolist(),  (phi_mat_V.abs() < 2e-2).sum())
            n_prune += self.decomposer.prunePhases(phi_mat_U, epsilon=epsilon)
            n_prune += self.decomposer.prunePhases(phi_mat_V, epsilon=epsilon)
            U_recon = self.decomposer.reconstruct(delta_list_U, phi_mat_U)
            V_recon = self.decomposer.reconstruct(delta_list_V, phi_mat_V)
            W_recon = torch.mm(U_recon, torch.mm(Sigma, V_recon))
            self.layers[layer].weight.data = W_recon
        return n_prune

    def restoreWeights(self):
        for layer in self.layers:
            self.layers[layer].weight.data.copy_(self.weights_backup[layer])
        print(f"[I] Weights restored from backup")

    def forward(self, x):
        x = x[..., ::2, ::2].contiguous()
        x = x.view(-1, self.n_feat)
        x = torch.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        return F.log_softmax(self.fc2(x), dim=1)


class MLP(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, device=torch.device("cuda")):
        super(MLP, self).__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.device = device
        self.fc1 = CirculantLinear(n_feat, 60, 4, device=device)
        self.fc2 = CirculantLinear(60, n_class, 10, device=device)
        # self.fc3 = CirculantLinear(400, n_class, 10, device=device)
        # self.fc1 = nn.Linear(28*28,32)
        # self.fc2 = nn.Linear(32,20)
        # self.fc3 = nn.Linear(20,10)
        self.init_weights()

    def init_weights(self):
        self.fc1.init_weights()
        self.fc2.init_weights()
        # self.fc3.init_weights()
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)

    def forward(self, x):
        x = x.view(-1, self.n_feat)
        x = torch.sigmoid(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        return F.log_softmax(self.fc2(x), dim=1)


class SparseMLP(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, threshold=0, l1_fraction=0, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.threshold = threshold
        self.l1_fraction = l1_fraction
        self.device = device

        self.fc1 = CirculantLinear_v2(n_feat, 120, 8, device=device)
        self.fc1.createMask(drop_rate=0.2)
        self.fc2 = CirculantLinear_v2(120, n_class, 10, device=device)
        # self.fc2.createMask(drop_rate=0.2)
        # self.fc3 = CirculantLinear(400, n_class, 10, device=device)
        # self.fc1 = nn.Linear(28*28,32)
        # self.fc2 = nn.Linear(32,20)
        # self.fc3 = nn.Linear(20,10)
        self.init_weights()

    def init_weights(self):
        self.fc1.init_weights()
        self.fc2.init_weights()
        # self.fc3.init_weights()
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)

    def getLassoLoss(self):
        lasso_loss = self.fc1.getGroupLassoRegularizer(threshold=self.threshold, l1_fraction=self.l1_fraction) + \
            self.fc2.getGroupLassoRegularizer(
                threshold=self.threshold, l1_fraction=self.l1_fraction)
        return lasso_loss

    def forward(self, x):
        # x = x[:, :, ::2, ::2].contiguous()

        x = x.view(-1, self.n_feat)
        x = torch.sigmoid(self.fc1(x))
        out = F.log_softmax(self.fc2(x), dim=1)
        # lasso_loss = self.getLassoLoss()
        return out  # , lasso_loss


class LassoMLP(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, threshold=0, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.threshold = threshold

        self.device = device

        self.fc1 = CirculantLinear_v3(
            n_feat, 120, 8, group_lasso=True, device=device)
        self.fc2 = CirculantLinear_v3(
            120, n_class, 10, group_lasso=True, device=device)
        self.layers = {"fc1": self.fc1, "fc2": self.fc2}

        self.init_weights()

    def init_weights(self):
        self.fc1.init_weights()
        self.fc2.init_weights()

        # self.fc3.init_weights()
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)

    def getLassoLoss(self):
        group_lasso = []
        group_lasso.append(self.fc1.getGroupLasso())
        group_lasso.append(self.fc2.getGroupLasso())
        group_lasso = torch.cat([i.contiguous().view(-1)
                                 for i in group_lasso], 0)
        lasso_loss = torch.mean(group_lasso)

        return lasso_loss

    def getDropMask(self, drop_thres=0.01):
        group_lasso = {
            layer: self.layers[layer].getGroupLasso() for layer in self.layers}
        assert drop_thres >= 0, "[E] Drop threshold must be non-negative a real number"
        self.drop_masks = {layer: lasso >=
                           drop_thres for layer, lasso in group_lasso.items()}

        return self.drop_masks

    def applyDropMask(self, drop_masks=None):
        if(not drop_masks):
            return
        for layer, mask in drop_masks.items():

            self.layers[layer].eigens.data = self.layers[layer].eigens.data * \
                mask.to(torch.float32).unsqueeze(2)
        # print("Drop mask applied to", list(drop_masks.keys()))

    def forward(self, x):
        # x = x[:, :, ::2, ::2].contiguous()

        x = x.view(-1, self.n_feat)
        x = torch.relu(self.fc1(x))
        out = F.log_softmax(self.fc2(x), dim=1)
        lasso_loss = self.getLassoLoss()
        return out, lasso_loss


class LassoMLP_v2(nn.Module):
    def __init__(self, n_feat=28 * 28, n_class=10, threshold=0, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.threshold = threshold

        self.device = device

        self.fc1 = CirculantLinear_v4(
            n_feat, 120, 8, group_lasso=True, device=device)
        self.fc2 = CirculantLinear_v4(
            120, n_class, 10, group_lasso=True, device=device)
        self.layers = {"fc1": self.fc1, "fc2": self.fc2}
        self.drop_masks = None

        self.init_weights()

    def init_weights(self):
        self.fc1.init_weights()
        self.fc2.init_weights()

        # self.fc3.init_weights()
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)

    def getLassoLoss(self):
        group_lasso = []
        group_lasso.append(self.fc1.getGroupLasso())
        group_lasso.append(self.fc2.getGroupLasso())
        group_lasso = torch.cat([i.contiguous().view(-1)
                                 for i in group_lasso], 0)
        lasso_loss = torch.mean(group_lasso)

        return lasso_loss

    def getDropMask(self, drop_thres=0.01):
        group_lasso = {
            layer: self.layers[layer].getGroupLasso() for layer in self.layers}
        assert drop_thres >= 0, "[E] Drop threshold must be non-negative a real number"
        self.drop_masks = {layer: lasso >=
                           drop_thres for layer, lasso in group_lasso.items()}

        return self.drop_masks

    def applyDropMask(self, drop_masks=None):
        if(not drop_masks):
            return
        for layer, mask in drop_masks.items():

            self.layers[layer].eigens.data = self.layers[layer].eigens.data * \
                mask.to(torch.float32).unsqueeze(2)
        # print("Drop mask applied to", list(drop_masks.keys()))

    def getAttenuatorPenalty(self, drop_masks=None):
        '''Masked eigens should not contribute to this penalty
        '''
        penalty = {layer: self.layers[layer].getAttenuatorPenalty()
                   for layer in self.layers}
        if(drop_masks is not None):
            penalty = torch.mean(torch.cat([(penalty[layer] * drop_masks[layer].to(
                torch.float32).unsqueeze(2)).contiguous().view(-1) for layer in self.layers], dim=0))
        else:
            penalty = torch.mean(torch.cat(
                [penalty[layer].contiguous().view(-1) for layer in self.layers], dim=0))
        return penalty

    def forward(self, x):
        # x = x[:, :, ::2, ::2].contiguous()

        x = x.view(-1, self.n_feat)
        x = torch.relu(self.fc1(x))
        out = F.log_softmax(self.fc2(x), dim=1)
        lasso_loss = self.getLassoLoss()
        attenuator_penalty = self.getAttenuatorPenalty(
            drop_masks=self.drop_masks)
        return out, lasso_loss, attenuator_penalty


class LassoMLP_v3(nn.Module):
    '''v3: add phase drift
    '''

    def __init__(self, n_feat=28 * 28, n_class=10, threshold=0, group_lasso=True, phase_drift=False, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.threshold = threshold

        self.device = device

        self.fc1 = CirculantLinear_v6(
            n_feat, 1024, 8, group_lasso=group_lasso, phase_drift=phase_drift, phase_drift_std=0.15, device=device)
        self.fc2 = CirculantLinear_v6(
            1024, n_class, 2, group_lasso=group_lasso, phase_drift=phase_drift, phase_drift_std=0.15, device=device)
        self.layers = {"fc1": self.fc1, "fc2": self.fc2}
        self.drop_masks = None

        self.init_weights()
        # Will do activation in electronics. So, use activation that has high performance.
        # self.modrelu1 = ModReLU(bias_shape=120, device=self.device)
        self.act1 = nn.ReLU()

    def init_weights(self):
        self.fc1.init_weights()
        self.fc2.init_weights()

        # self.fc3.init_weights()
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)

    def getLassoLoss(self, sparsity=None):
        group_lasso = []
        group_lasso.append(self.fc1.getGroupLasso())
        group_lasso.append(self.fc2.getGroupLasso())
        group_lasso = torch.cat([i.contiguous().view(-1)
                                 for i in group_lasso], 0)

        lasso_loss = torch.mean(group_lasso)

        return lasso_loss

    def getDropMask(self, drop_thres=0.01):
        group_lasso = {
            layer: self.layers[layer].getGroupLasso() for layer in self.layers}
        assert drop_thres >= 0, "[E] Drop threshold must be non-negative a real number"
        self.drop_masks = {layer: lasso >=
                           drop_thres for layer, lasso in group_lasso.items()}

        return self.drop_masks

    def applyDropMask(self, drop_masks=None):
        if(not drop_masks):
            return
        for layer, mask in drop_masks.items():

            self.layers[layer].eigens.data = self.layers[layer].eigens.data * \
                mask.to(torch.float32).unsqueeze(2)
        # print("Drop mask applied to", list(drop_masks.keys()))

    def getAttenuatorPenalty(self, drop_masks=None, sparsity=None):
        '''Masked eigens should not contribute to this penalty
        '''
        penalty = {layer: self.layers[layer].getAttenuatorPenalty()
                   for layer in self.layers}
        if(drop_masks is not None):
            penalty = torch.sqrt(torch.mean(torch.cat([(penalty[layer] * drop_masks[layer].to(
                torch.float32).unsqueeze(2)).contiguous().view(-1) for layer in self.layers], dim=0)) / (1 - sparsity))
            # penalty = torch.sqrt(torch.mean(torch.cat([(penalty[layer] * drop_masks[layer].data.to(
            #     torch.float32).unsqueeze(2)).contiguous().view(-1) for layer in self.layers], dim=0)))
        else:
            penalty = torch.sqrt(torch.mean(torch.cat(
                [penalty[layer].contiguous().view(-1) for layer in self.layers], dim=0)))
        return penalty

    def forward(self, x):
        # x = x[..., ::2, ::2].contiguous()

        x = x.view(-1, self.n_feat)
        x = real_to_complex(x)  # expand it to a complex vector

        x = self.fc1(x)
        x = self.act1(x)
        # x = self.modrelu1(x)
        x = self.fc2(x)
        x = get_complex_magnitude(x)  # photodetector will detect magnitude ?

        out = F.log_softmax(x, dim=1)
        return out


class LassoMLP_v4(nn.Module):
    '''v3: real number computation without phase drift
    '''

    def __init__(self, n_feat=28 * 28, n_class=10, threshold=0, group_lasso=True, phase_drift=False, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.threshold = threshold

        self.device = device

        self.fc1 = CirculantLinear_v5(
            n_feat, 1024, 8, group_lasso=group_lasso, device=device)
        self.fc2 = CirculantLinear_v5(
            1024, n_class, 2, group_lasso=group_lasso, device=device)
        self.layers = {"fc1": self.fc1, "fc2": self.fc2}
        self.drop_masks = None

        self.init_weights()
        # Will do activation in electronics. So, use activation that has high performance.
        # self.modrelu1 = ModReLU(bias_shape=120, device=self.device)
        self.act1 = nn.ReLU()

    def init_weights(self):
        self.fc1.init_weights()
        self.fc2.init_weights()

        # self.fc3.init_weights()
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)

    def getLassoLoss(self, sparsity=None):
        group_lasso = []
        group_lasso.append(self.fc1.getGroupLasso())
        group_lasso.append(self.fc2.getGroupLasso())
        group_lasso = torch.cat([i.contiguous().view(-1)
                                 for i in group_lasso], 0)

        lasso_loss = torch.mean(group_lasso)

        return lasso_loss

    def getDropMask(self, drop_thres=0.01):
        group_lasso = {
            layer: self.layers[layer].getGroupLasso() for layer in self.layers}
        assert drop_thres >= 0, "[E] Drop threshold must be non-negative a real number"
        self.drop_masks = {layer: lasso >=
                           drop_thres for layer, lasso in group_lasso.items()}

        return self.drop_masks

    def applyDropMask(self, drop_masks=None):
        if(not drop_masks):
            return
        for layer, mask in drop_masks.items():

            self.layers[layer].eigens.data = self.layers[layer].eigens.data * \
                mask.to(torch.float32).unsqueeze(2)
        # print("Drop mask applied to", list(drop_masks.keys()))

    def getAttenuatorPenalty(self, drop_masks=None, sparsity=None):
        '''Masked eigens should not contribute to this penalty
        '''
        penalty = {layer: self.layers[layer].getAttenuatorPenalty()
                   for layer in self.layers}
        if(drop_masks is not None):
            penalty = torch.sqrt(torch.mean(torch.cat([(penalty[layer] * drop_masks[layer].to(
                torch.float32).unsqueeze(2)).contiguous().view(-1) for layer in self.layers], dim=0)) / (1 - sparsity))
            # penalty = torch.sqrt(torch.mean(torch.cat([(penalty[layer] * drop_masks[layer].data.to(
            #     torch.float32).unsqueeze(2)).contiguous().view(-1) for layer in self.layers], dim=0)))
        else:
            penalty = torch.sqrt(torch.mean(torch.cat(
                [penalty[layer].contiguous().view(-1) for layer in self.layers], dim=0)))
        return penalty

    def forward(self, x):
        # x = x[..., ::2, ::2].contiguous()

        x = x.view(-1, self.n_feat)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        return out


class LassoMLP_v3_3Layer(nn.Module):
    '''v3: add phase drift
    '''

    def __init__(self, n_feat=28 * 28, n_class=10, threshold=0, group_lasso=True, phase_drift=False, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.threshold = threshold

        self.device = device

        self.fc1 = CirculantLinear_v6(
            n_feat, 60, 4, group_lasso=group_lasso, phase_drift=phase_drift, phase_drift_std=0.15, device=device)
        self.fc2 = CirculantLinear_v6(
            60, 32, 4, group_lasso=group_lasso, phase_drift=phase_drift, phase_drift_std=0.15, device=device)
        self.fc3 = CirculantLinear_v6(
            32, n_class, 2, group_lasso=group_lasso, phase_drift=phase_drift, phase_drift_std=0.15, device=device)
        self.layers = {"fc1": self.fc1, "fc2": self.fc2, "fc3": self.fc3}
        self.drop_masks = None

        self.init_weights()
        # Will do activation in electronics. So, use activation that has high performance.
        # self.modrelu1 = ModReLU(bias_shape=120, device=self.device)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

    def init_weights(self):
        self.fc1.init_weights()
        self.fc2.init_weights()
        self.fc3.init_weights()

    def getLassoLoss(self, sparsity=None):
        group_lasso = []
        group_lasso.append(self.fc1.getGroupLasso())
        group_lasso.append(self.fc2.getGroupLasso())
        group_lasso.append(self.fc3.getGroupLasso())
        group_lasso = torch.cat([i.contiguous().view(-1)
                                 for i in group_lasso], 0)

        lasso_loss = torch.mean(group_lasso)

        return lasso_loss

    def getDropMask(self, drop_thres=0.01):
        group_lasso = {
            layer: self.layers[layer].getGroupLasso() for layer in self.layers}
        assert drop_thres >= 0, "[E] Drop threshold must be non-negative a real number"
        self.drop_masks = {layer: lasso >=
                           drop_thres for layer, lasso in group_lasso.items()}

        return self.drop_masks

    def applyDropMask(self, drop_masks=None):
        if(not drop_masks):
            return
        for layer, mask in drop_masks.items():

            self.layers[layer].eigens.data = self.layers[layer].eigens.data * \
                mask.to(torch.float32).unsqueeze(2)
        # print("Drop mask applied to", list(drop_masks.keys()))

    def getAttenuatorPenalty(self, drop_masks=None, sparsity=None):
        '''Masked eigens should not contribute to this penalty
        '''
        penalty = {layer: self.layers[layer].getAttenuatorPenalty()
                   for layer in self.layers}
        if(drop_masks is not None):
            penalty = torch.sqrt(torch.mean(torch.cat([(penalty[layer] * drop_masks[layer].to(
                torch.float32).unsqueeze(2)).contiguous().view(-1) for layer in self.layers], dim=0)) / (1 - sparsity))
            # penalty = torch.sqrt(torch.mean(torch.cat([(penalty[layer] * drop_masks[layer].data.to(
            #     torch.float32).unsqueeze(2)).contiguous().view(-1) for layer in self.layers], dim=0)))
        else:
            penalty = torch.sqrt(torch.mean(torch.cat(
                [penalty[layer].contiguous().view(-1) for layer in self.layers], dim=0)))
        return penalty

    def forward(self, x):
        x = x[..., ::2, ::2].contiguous()

        x = x.view(-1, self.n_feat)
        x = real_to_complex(x)  # expand it to a complex vector

        x = self.fc1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.act2(x)

        x = self.fc3(x)
        x = get_complex_magnitude(x)  # photodetector will detect magnitude ?
        out = F.log_softmax(x, dim=1)
        return out


class LassoMLP_v3_Tiny(nn.Module):
    '''v3: add phase drift
    '''

    def __init__(self, n_feat=8, n_class=4, threshold=0, group_lasso=True, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.threshold = threshold

        self.device = device

        self.fc1 = CirculantLinear_v5(
            n_feat, 32, 8, group_lasso=group_lasso, device=device)
        self.fc2 = CirculantLinear_v5(
            32, n_class, 2, group_lasso=group_lasso, device=device)
        self.layers = {"fc1": self.fc1, "fc2": self.fc2}
        self.drop_masks = None

        self.init_weights()
        # Will do activation in electronics. So, use activation that has high performance.
        # self.modrelu1 = ModReLU(bias_shape=120, device=self.device)
        self.act1 = nn.ReLU()

    def init_weights(self):
        self.fc1.init_weights()
        self.fc2.init_weights()

        # self.fc3.init_weights()
        # nn.init.xavier_normal_(self.fc1.weight)
        # nn.init.xavier_normal_(self.fc2.weight)
        # nn.init.xavier_normal_(self.fc3.weight)

    def getLassoLoss(self, sparsity=None):
        group_lasso = []
        group_lasso.append(self.fc1.getGroupLasso())
        group_lasso.append(self.fc2.getGroupLasso())
        group_lasso = torch.cat([i.contiguous().view(-1)
                                 for i in group_lasso], 0)

        lasso_loss = torch.mean(group_lasso)
        if(sparsity is not None):
            lasso_loss = lasso_loss  # / (1 - sparsity)

        return lasso_loss

    def getDropMask(self, drop_thres=0.01):
        group_lasso = {
            layer: self.layers[layer].getGroupLasso() for layer in self.layers}
        assert drop_thres >= 0, "[E] Drop threshold must be non-negative a real number"
        self.drop_masks = {layer: lasso >=
                           drop_thres for layer, lasso in group_lasso.items()}

        return self.drop_masks

    def applyDropMask(self, drop_masks=None):
        if(not drop_masks):
            return
        for layer, mask in drop_masks.items():

            self.layers[layer].eigens.data = self.layers[layer].eigens.data * \
                mask.to(torch.float32).unsqueeze(2)
        # print("Drop mask applied to", list(drop_masks.keys()))

    def getAttenuatorPenalty(self, drop_masks=None, sparsity=None):
        '''Masked eigens should not contribute to this penalty
        '''
        penalty = {layer: self.layers[layer].getAttenuatorPenalty()
                   for layer in self.layers}
        if(drop_masks is not None):
            # penalty = torch.sqrt(torch.mean(torch.cat([(penalty[layer] * drop_masks[layer].to(
            #     torch.float32).unsqueeze(2)).contiguous().view(-1) for layer in self.layers], dim=0))/(1-sparsity))
            penalty = torch.sqrt(torch.mean(torch.cat([(penalty[layer] * drop_masks[layer].data.to(
                torch.float32).unsqueeze(2)).contiguous().view(-1) for layer in self.layers], dim=0)))
        else:
            penalty = torch.sqrt(torch.mean(torch.cat(
                [penalty[layer].contiguous().view(-1) for layer in self.layers], dim=0)))
        return penalty

    def forward(self, x):

        x = x.view(-1, self.n_feat)
        # x = real_to_complex(x)  # expand it to a complex vector

        x = self.fc1(x)
        x = self.act1(x)
        # x = self.modrelu1(x)
        x = self.fc2(x)
        # x = get_complex_magnitude(x)  # photodetector will detect magnitude ?
        out = F.log_softmax(x, dim=1)
        return out


class FCMLP_Tiny(nn.Module):
    def __init__(self, n_feat=8, n_class=4, device=torch.device("cuda")):
        super().__init__()
        self.n_feat = n_feat
        self.n_class = n_class
        self.device = device
        self.fc1 = nn.Linear(n_feat, 8, bias=False)
        self.fc2 = nn.Linear(8, n_class, bias=False)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):

        x = x.view(-1, self.n_feat)
        x = torch.relu(self.fc1(x))
        # x = torch.sigmoid(self.fc2(x))
        return F.log_softmax(self.fc2(x), dim=1)



class HP_CLASS_CNN(nn.Module):
    def __init__(self, img_height, img_width, in_channels, n_class, kernel_list=[16], hidden_list=[32], pool_out_size=5, in_bits=32, w_bits=32, act="relu", act_thres=6, mode="oconv", device=torch.device("cuda")):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.n_class = n_class
        self.kernel_list = kernel_list
        self.hidden_list = hidden_list
        self.pool_out_size = pool_out_size
        self.in_bits = in_bits
        self.w_bits = w_bits
        self.act = act
        self.act_thres = act_thres
        self.mode = mode
        self.device = device
        self.build_layers()
        self.reset_parameters()

    def build_layers(self):
        self.conv_layers = OrderedDict()
        self.bn_layers = OrderedDict()
        self.fc_layers = OrderedDict()
        self.acts = OrderedDict()
        for idx, out_channels in enumerate(self.kernel_list, 0):
            layer_name = "conv" + str(idx+1)
            bn_name = "bn" + str(idx+1)
            act_name = "conv_act" + str(idx+1)
            in_channels = self.in_channels if(
                idx == 0) else self.kernel_list[idx-1]
            conv = OAdder2d_Q(in_channels,
                            out_channels,
                            3,
                            stride=1,
                            padding=1,
                            in_bit=self.in_bits,
                            w_bit=self.w_bits,
                            mode=self.mode,
                            bias=False,
                            device=self.device)

            bn = nn.BatchNorm2d(out_channels)
            # activation = BiasReLU(bias_shape=(out_channels, 1, 1),
            #                       max_val=self.act_thres,
            #                       device=self.device)
            # activation = ReLUN(self.act_thres, inplace=True)
            if(self.act == "relu"):
                activation = nn.ReLU(inplace=True)
            elif(self.act == "relun"):
                activation = ReLUN(self.act_thres, inplace=True)
            elif(self.act == "tanh"):
                activation = nn.Tanh()
            else:
                activation = nn.Identity()

            self.conv_layers[layer_name] = conv
            self.bn_layers[layer_name] = bn
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, conv)
            super().__setattr__(bn_name, bn)
            super().__setattr__(act_name, activation)

        self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)

        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            act_name = "fc_act" + str(idx+1)
            in_channel = self.kernel_list[-1]*self.pool_out_size * \
                self.pool_out_size if idx == 0 else self.hidden_list[idx-1]
            out_channel = hidden_dim
            fc = nn.Linear(in_channel, out_channel, bias=False)
            activation = nn.ReLU()
            self.fc_layers[layer_name] = fc
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            super().__setattr__(act_name, activation)

        layer_name = "fc"+str(len(self.hidden_list)+1)
        fc = nn.Linear(self.hidden_list[-1], self.n_class, bias=False)
        super().__setattr__(layer_name, fc)
        self.fc_layers[layer_name] = fc
        # self.reg = nn.ModuleList(
        #     list(self.conv_layers.values()) + list(self.fc_layers.values()))

    def reset_parameters(self, initializer=nn.init.kaiming_normal_):
        # for layer in self.conv_layers:
        #     # print(self.layers[layer])
        #     self.conv_layers[layer].reset_parameters()
        for layer in self.fc_layers:
            nn.init.kaiming_normal_(self.fc_layers[layer].weight)

    def enable_calibration(self):
        for layer in self.conv_layers:
            self.conv_layers[layer].enable_calibration()

    def disable_calibration(self):
        for layer in self.conv_layers:
            self.conv_layers[layer].disable_calibration()

    def static_pre_calibration(self):
        for layer in self.conv_layers:
            self.conv_layers[layer].static_pre_calibration()

    def assign_engines(self, out_par=1, image_par=1, phase_noise_std=0, disk_noise_std=0, deterministic=False):
        self.phase_noise_std = phase_noise_std
        self.disk_noise_std = disk_noise_std
        for layer in self.conv_layers:
            self.conv_layers[layer].assign_engines(out_par, image_par, phase_noise_std, disk_noise_std, deterministic)

    def deassign_engines(self):
        for layer in self.conv_layers:
            self.conv_layers[layer].deassign_engines()

    def forward(self, x):
        for idx, layer in enumerate(self.conv_layers, 1):
            x = self.conv_layers[layer](x)
            # if(idx == 2):
            #     print("conv")
            #     print_stat(x)
            x = self.bn_layers[layer](x)
            # if(idx == 2):
            #     print("bn")
            #     print_stat(x)
            x = self.acts[layer](x)
        n_fc = len(self.fc_layers)

        x = self.pool2d(x)

        x = x.contiguous().view(-1, x.size(1)*x.size(2)*x.size(3))

        for idx, layer in enumerate(self.fc_layers, 1):

            x = self.fc_layers[layer](x)

            if(idx < n_fc):
                x = self.acts[layer](x)

        out = F.log_softmax(x, dim=1)
        return out



class HP_CLASS_CNN2(nn.Module):
    def __init__(self, img_height, img_width, in_channels, n_class, kernel_list=[16], hidden_list=[32], pool_out_size=5, in_bits=32, w_bits=32, act="relu", act_thres=6, mode="oconv", input_augment=False, device=torch.device("cuda")):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.n_class = n_class
        self.kernel_list = kernel_list
        self.hidden_list = hidden_list
        self.pool_out_size = pool_out_size
        self.in_bits = in_bits
        self.w_bits = w_bits
        self.act = act
        self.act_thres = act_thres
        self.mode = mode
        self.input_augment = input_augment

        self.device = device
        self.build_layers()
        self.reset_parameters()

    def build_layers(self):
        self.conv_layers = OrderedDict()
        self.bn_layers = OrderedDict()
        self.fc_layers = OrderedDict()
        self.acts = OrderedDict()
        for idx, out_channels in enumerate(self.kernel_list, 0):
            layer_name = "conv" + str(idx+1)
            bn_name = "bn" + str(idx+1)
            act_name = "conv_act" + str(idx+1)
            in_channels = self.in_channels if(
                idx == 0) else self.kernel_list[idx-1]
            conv = OAdder2d_Q(in_channels,
                            out_channels,
                            3,
                            stride=1,
                            padding=1,
                            in_bit=self.in_bits,
                            w_bit=self.w_bits,
                            mode=self.mode,
                            bias=False,
                            input_augment=self.input_augment,
                            device=self.device)

            bn =  nn.BatchNorm2d(out_channels)
            # activation = BiasReLU(bias_shape=(out_channels, 1, 1),
            #                       max_val=self.act_thres,
            #                       device=self.device)
            # activation = ReLUN(self.act_thres, inplace=True)
            if(self.act == "relu"):
                activation = nn.ReLU(inplace=True)
            elif(self.act == "relun"):
                activation = ReLUN(self.act_thres, inplace=True)
            elif(self.act == "tanh"):
                activation = nn.Tanh()
            else:
                activation = nn.Identity()

            self.conv_layers[layer_name] = conv
            self.bn_layers[layer_name] = bn
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, conv)
            super().__setattr__(bn_name, bn)
            super().__setattr__(act_name, activation)

        self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)

        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            act_name = "fc_act" + str(idx+1)
            in_channel = self.kernel_list[-1]*self.pool_out_size * \
                self.pool_out_size if idx == 0 else self.hidden_list[idx-1]
            out_channel = hidden_dim
            # fc = nn.Linear(in_channel, out_channel, bias=False)
            fc = OLinear_Q(in_channel, out_channel, self.in_bits, self.w_bits, False, self.mode, self.input_augment, self.device)
            if(self.act == "relu"):
                activation = nn.ReLU(inplace=True)
            elif(self.act == "relun"):
                activation = ReLUN(self.act_thres, inplace=True)
            elif(self.act == "tanh"):
                activation = nn.Tanh()
            else:
                activation = nn.Identity()
            self.fc_layers[layer_name] = fc
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            super().__setattr__(act_name, activation)

        layer_name = "fc"+str(len(self.hidden_list)+1)
        fc = OLinear_Q(self.hidden_list[-1], self.n_class, self.in_bits, self.w_bits, False, self.mode, self.input_augment, self.device)

        # fc = nn.Linear(self.hidden_list[-1], self.n_class, bias=False)
        super().__setattr__(layer_name, fc)
        self.fc_layers[layer_name] = fc
        # self.reg = nn.ModuleList(
        #     list(self.conv_layers.values()) + list(self.fc_layers.values()))

    def reset_parameters(self, initializer=nn.init.kaiming_normal_):
        # for layer in self.conv_layers:
        #     # print(self.layers[layer])
        #     self.conv_layers[layer].reset_parameters()
        # for layer in self.fc_layers:
        #     nn.init.kaiming_normal_(self.fc_layers[layer].weight)
        pass

    def init_lagrangian_lambda(self):
        self.lagrangian_lambda = OrderedDict()
        # for layer in self.conv_layers:
        #     self.lagrangian_lambda[layer] = torch.ones(self.conv_layers[layer].output_channel).to(self.device)-0.9999
        for layer in self.conv_layers:
            self.lagrangian_lambda[layer] = torch.ones(self.conv_layers[layer].input_channel).to(self.device)-0.999
        for layer in self.fc_layers:
            self.lagrangian_lambda[layer] = torch.ones(self.fc_layers[layer].input_channel).to(self.device)-0.999

    def update_lagrangian_lambda(self, learning_rate):
        for layer in self.conv_layers:
            self.lagrangian_lambda[layer] += 2e-2 * self.conv_layers[layer].beta.data.squeeze()
        for layer in self.fc_layers:
            self.lagrangian_lambda[layer] += 2e-2 * self.fc_layers[layer].beta.data.squeeze()

    def clamp_alpha(self):
        for layer in self.conv_layers:
            self.conv_layers[layer].beta.data.clamp_(0, 1)
        for layer in self.fc_layers:
            self.fc_layers[layer].beta.data.clamp_(0, 1)

    def set_alpha(self, alpha):
        for layer in self.conv_layers:
            self.conv_layers[layer].beta.data[...] = alpha
        for layer in self.fc_layers:
            self.fc_layers[layer].beta.data[...] = alpha

    def get_alpha_loss(self):
        # loss = None
        # for layer in self.conv_layers:
        #     loss = torch.dot(self.lagrangian_lambda[layer], self.conv_layers[layer].beta.squeeze(0).squeeze(-1).squeeze(-1)) if loss is None else loss + torch.dot(self.lagrangian_lambda[layer], self.conv_layers[layer].beta.squeeze(0).squeeze(-1).squeeze(-1))
        # for layer in self.fc_layers:
        #     loss = torch.dot(self.lagrangian_lambda[layer], self.fc_layers[layer].beta.squeeze(0)) if loss is None else loss + torch.dot(self.lagrangian_lambda[layer], self.fc_layers[layer].beta.squeeze(0))
        # if(loss is None):
        #     return torch.Tensor([0])
        loss = 0
        mu = 1e-2
        for layer in self.conv_layers:
            beta = self.conv_layers[layer].beta.squeeze(0).squeeze(-1).squeeze(-1)
            # loss = loss + torch.dot(self.lagrangian_lambda[layer], beta ) + mu*(beta*beta).sum()
            loss = loss + torch.dot(self.lagrangian_lambda[layer], beta + mu * beta*beta )
        for layer in self.fc_layers:
            beta = self.fc_layers[layer].beta.squeeze(0)
            # loss = loss + torch.dot(self.lagrangian_lambda[layer], beta) + mu * (beta*beta).sum()
            loss = loss + torch.dot(self.lagrangian_lambda[layer], beta + mu * beta*beta)
        if(loss is None):
            return torch.Tensor([0])
        return loss

    def get_average_alpha(self):
        alpha = []
        for layer in self.conv_layers:
            alpha.append(self.conv_layers[layer].beta.data.squeeze(0).squeeze(-1).squeeze(-1))
        for layer in self.fc_layers:
            alpha.append(self.fc_layers[layer].beta.data.squeeze(0))
        alpha = torch.cat(alpha, dim=0)
        return alpha.mean().data.item()

    def enable_trainable_alpha(self):
        for layer in self.conv_layers:
            self.conv_layers[layer].beta.requires_grad = True
        for layer in self.fc_layers:
            self.fc_layers[layer].beta.requires_grad = True

    def disable_trainable_alpha(self):
        try:
            for layer in self.conv_layers:
                self.conv_layers[layer].beta.requires_grad = False
            for layer in self.fc_layers:
                self.fc_layers[layer].beta.requires_grad = False
        except:
            pass

    def enable_input_augment(self):
        for layer in self.conv_layers:
            self.conv_layers[layer].input_augment = True
        for layer in self.fc_layers:
            self.fc_layers[layer].input_augment = True

    def disable_input_augment(self):
        for layer in self.conv_layers:
            self.conv_layers[layer].input_augment = False
        for layer in self.fc_layers:
            self.fc_layers[layer].input_augment = False

    def enable_calibration(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].enable_calibration()
        for layer in self.conv_layers:
            self.conv_layers[layer].enable_calibration()

    def disable_calibration(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].disable_calibration()
        for layer in self.conv_layers:
            self.conv_layers[layer].disable_calibration()

    def static_pre_calibration(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].static_pre_calibration()
        for layer in self.conv_layers:
            self.conv_layers[layer].static_pre_calibration()

    def assign_engines(self, out_par=1, image_par=1, phase_noise_std=0, disk_noise_std=0, deterministic=False):
        self.phase_noise_std = phase_noise_std
        self.disk_noise_std = disk_noise_std
        for layer in self.fc_layers:
            self.fc_layers[layer].assign_engines(out_par, image_par, phase_noise_std, disk_noise_std, deterministic)
        for layer in self.conv_layers:
            self.conv_layers[layer].assign_engines(out_par, image_par, phase_noise_std, disk_noise_std, deterministic)

    def deassign_engines(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].deassign_engines()
        for layer in self.conv_layers:
            self.conv_layers[layer].deassign_engines()

    def robust_reassign_engines(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].robust_reassign_engines()
        for layer in self.conv_layers:
            self.conv_layers[layer].robust_reassign_engines()

    def robust_reassign_engines_fine(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].robust_reassign_engines_fine()
        for layer in self.conv_layers:
            self.conv_layers[layer].robust_reassign_engines_fine()

    def assign_ring_noise(self, ring_noise_std=0, ring_crosstalk_perc=0, deterministic=False):
        for layer in self.fc_layers:
            self.fc_layers[layer].assign_ring_noise(ring_noise_std, ring_crosstalk_perc, deterministic)
        for layer in self.conv_layers:
            self.conv_layers[layer].assign_ring_noise(ring_noise_std, ring_crosstalk_perc, deterministic)

    def mzi_quantize(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].mzi_quantize()
        for layer in self.conv_layers:
            self.conv_layers[layer].mzi_quantize()

    def apply_unitary_projection(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].U.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(
                self.fc_layers[layer].U.data.detach().cpu().numpy())).to(self.device))
            self.fc_layers[layer].V.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(
                self.fc_layers[layer].V.data.detach().cpu().numpy())).to(self.device))
        for layer in self.conv_layers:
            self.conv_layers[layer].U.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(
                self.conv_layers[layer].U.data.detach().cpu().numpy())).to(self.device))
            self.conv_layers[layer].V.data.copy_(torch.from_numpy(projection_matrix_to_unitary_cpu(
                self.conv_layers[layer].V.data.detach().cpu().numpy())).to(self.device))

    def init_act_distrib_loss(self):
        self.act_distrib_loss = 0

    def register_act_distrib_loss(self, act):
        act = act.clone()
        k1, k2, k3 = 1, 0.25, 0.25
        if(act.dim() == 4):
            act_mean = act.mean(dim=(0,2,3), keepdim=True)
            act_std = (((act - act_mean)**2).mean(dim=(0,2,3), keepdim=True) + 1e-10).sqrt()
        elif(act.dim() == 2):
            act_mean = act.mean(dim=0, keepdim=True)
            act_std = (((act - act_mean)**2).mean(dim=0, keepdim=True) + 1e-10).sqrt()
        Ld = ((act_mean.abs() - (k1 * act_std+0.5)).clamp(min=0)**2).mean()
        Ls = ((k2 * act_std - 1).clamp(min=0)**2).mean()
        Lm = (torch.min(1 - act_mean - k3*act_std, 1 + act_mean - k3*act_std).clamp(min=0) ** 2).mean()
        self.act_distrib_loss = self.act_distrib_loss + Ld + Ls + Lm

    def get_act_distrib_loss(self):
        return self.act_distrib_loss
        # loss = 0
        # k1, k2, k3 = 1, 0.25, 0.25
        # for act in self.activation_record:
        #     act_mean_abs = act.mean().abs()
        #     act_std = act.std()
        #     Ld = (act_mean_abs - k1 * act_std)**2
        #     Ls = (k2 * act_std - 1)**2
        #     Lm = (1 - act_mean_abs - k3 * act_std) ** 2
        #     loss = loss + Ld + Ls + Lm
        # return loss

    def forward(self, x, act_distrib_loss=False):
        if(act_distrib_loss):
            self.init_act_distrib_loss()
        for idx, layer in enumerate(self.conv_layers, 1):
            x = self.conv_layers[layer](x)
            x = self.bn_layers[layer](x)
            if(act_distrib_loss):
                self.register_act_distrib_loss(x)

            x = self.acts[layer](x)
        n_fc = len(self.fc_layers)

        x = self.pool2d(x)

        x = x.contiguous().view(-1, x.size(1)*x.size(2)*x.size(3))

        for idx, layer in enumerate(self.fc_layers, 1):

            x = self.fc_layers[layer](x)
            if(act_distrib_loss):
                self.register_act_distrib_loss(x)

            if(idx < n_fc):
                x = self.acts[layer](x)

        out = F.log_softmax(x, dim=1)

        return out



class HP_CLASS_CNN3(nn.Module):
    def __init__(self, img_height, img_width, in_channels, n_class, kernel_list=[16], hidden_list=[32], pool_out_size=5, in_bits=32, w_bits=32, act="relu", act_thres=6, mode="oconv", device=torch.device("cuda")):
        super().__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.in_channels = in_channels
        self.n_class = n_class
        self.kernel_list = kernel_list
        self.hidden_list = hidden_list
        self.pool_out_size = pool_out_size
        self.in_bits = in_bits
        self.w_bits = w_bits
        self.act = act
        self.act_thres = act_thres
        self.mode = mode
        self.device = device
        self.build_layers()
        self.reset_parameters()

    def build_layers(self):
        self.conv_layers = OrderedDict()
        self.bn_layers = OrderedDict()
        self.fc_layers = OrderedDict()
        self.acts = OrderedDict()
        for idx, out_channels in enumerate(self.kernel_list, 0):
            layer_name = "conv" + str(idx+1)
            bn_name = "bn" + str(idx+1)
            act_name = "conv_act" + str(idx+1)
            in_channels = self.in_channels if(
                idx == 0) else self.kernel_list[idx-1]
            conv = OAdder2d_Q(in_channels,
                            out_channels,
                            3,
                            stride=1,
                            padding=1,
                            in_bit=self.in_bits,
                            w_bit=self.w_bits,
                            mode=self.mode,
                            bias=False,
                            device=self.device)

            bn = nn.Identity()
            # activation = BiasReLU(bias_shape=(out_channels, 1, 1),
            #                       max_val=self.act_thres,
            #                       device=self.device)
            # activation = ReLUN(self.act_thres, inplace=True)
            if(self.act == "relu"):
                activation = nn.ReLU(inplace=True)
            elif(self.act == "relun"):
                activation = ReLUN(self.act_thres, inplace=True)
            elif(self.act == "tanh"):
                activation = nn.Tanh()
            else:
                activation = nn.Identity()
            activation = NormProp(out_channels, activation)

            self.conv_layers[layer_name] = conv
            self.bn_layers[layer_name] = bn
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, conv)
            super().__setattr__(bn_name, bn)
            super().__setattr__(act_name, activation)

        self.pool2d = nn.AdaptiveAvgPool2d(self.pool_out_size)

        for idx, hidden_dim in enumerate(self.hidden_list, 0):
            layer_name = "fc" + str(idx+1)
            act_name = "fc_act" + str(idx+1)
            in_channels = self.kernel_list[-1]*self.pool_out_size * \
                self.pool_out_size if idx == 0 else self.hidden_list[idx-1]
            out_channels = hidden_dim
            # fc = nn.Linear(in_channel, out_channel, bias=False)
            fc = OLinear_Q(in_channels, out_channels, self.in_bits, self.w_bits, False, self.device)
            if(self.act == "relu"):
                activation = nn.ReLU(inplace=True)
            elif(self.act == "relun"):
                activation = ReLUN(self.act_thres, inplace=True)
            elif(self.act == "tanh"):
                activation = nn.Tanh()
            else:
                activation = nn.Identity()
            activation = NormProp(out_channels, activation)
            self.fc_layers[layer_name] = fc
            self.acts[layer_name] = activation
            super().__setattr__(layer_name, fc)
            super().__setattr__(act_name, activation)

        layer_name = "fc"+str(len(self.hidden_list)+1)
        fc = OLinear_Q(self.hidden_list[-1], self.n_class, self.in_bits, self.w_bits, False, self.device)

        # fc = nn.Linear(self.hidden_list[-1], self.n_class, bias=False)
        super().__setattr__(layer_name, fc)
        self.fc_layers[layer_name] = fc
        # self.reg = nn.ModuleList(
        #     list(self.conv_layers.values()) + list(self.fc_layers.values()))

    def reset_parameters(self, initializer=nn.init.kaiming_normal_):
        # for layer in self.conv_layers:
        #     # print(self.layers[layer])
        #     self.conv_layers[layer].reset_parameters()
        # for layer in self.fc_layers:
        #     nn.init.kaiming_normal_(self.fc_layers[layer].weight)
        pass

    def enable_calibration(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].enable_calibration()
        for layer in self.conv_layers:
            self.conv_layers[layer].enable_calibration()

    def disable_calibration(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].disable_calibration()
        for layer in self.conv_layers:
            self.conv_layers[layer].disable_calibration()

    def static_pre_calibration(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].static_pre_calibration()
        for layer in self.conv_layers:
            self.conv_layers[layer].static_pre_calibration()

    def assign_engines(self, out_par=1, image_par=1, phase_noise_std=0, disk_noise_std=0, deterministic=False):
        self.phase_noise_std = phase_noise_std
        self.disk_noise_std = disk_noise_std
        for layer in self.fc_layers:
            self.fc_layers[layer].assign_engines(out_par, image_par, phase_noise_std, disk_noise_std, deterministic)
        for layer in self.conv_layers:
            self.conv_layers[layer].assign_engines(out_par, image_par, phase_noise_std, disk_noise_std, deterministic)

    def deassign_engines(self):
        for layer in self.fc_layers:
            self.fc_layers[layer].deassign_engines()
        for layer in self.conv_layers:
            self.conv_layers[layer].deassign_engines()

    def forward(self, x):
        for idx, layer in enumerate(self.conv_layers, 1):
            x = self.conv_layers[layer](x)
            # if(idx == 2):
            #     print("conv")
            #     print_stat(x)
            # x = self.bn_layers[layer](x)
            # if(idx == 2):
            #     print("bn")
            #     print_stat(x)
            x = self.acts[layer](x, self.conv_layers[layer].weight)
        n_fc = len(self.fc_layers)

        x = self.pool2d(x)

        x = x.contiguous().view(-1, x.size(1)*x.size(2)*x.size(3))

        for idx, layer in enumerate(self.fc_layers, 1):

            x = self.fc_layers[layer](x)

            if(idx < n_fc):
                x = self.acts[layer](x, self.fc_layers[layer].weight)

        out = F.log_softmax(x, dim=1)
        return out


if __name__ == "__main__":
    x = torch.Tensor([1, 2, 3]).requires_grad_(True)
    y = (x**3).sum()
    v = torch.tensor([1, 1, 1], dtype=torch.float32)
    g1 = grad(y, x, create_graph=True)[0]
    g2 = grad(g1.sum(), x, create_graph=True)[0]
    # g1 = x.grad
    print(g1)
    print(g2)

    exit(1)
    W = torch.ones(4, 3)
    device = torch.device("cuda")
    M, N = W.size(0), W.size(1)
    U, Sigma, V = np.linalg.svd(W.cpu().numpy(), full_matrices=True)
    U = torch.from_numpy(U).to(device)

    Sigma = np.diag(Sigma)
    print(Sigma)
    if(M > N):
        Sigma = np.concatenate([Sigma, np.zeros([M - N, N])], axis=0)
    elif(M < N):
        Sigma = np.concatenate([Sigma, np.zeros([M, N - M])], axis=1)
    Sigma = torch.from_numpy(Sigma).to(device).float()
    V = torch.from_numpy(V).to(device)
    print(U)
    print(Sigma)
    print(V)
    print(U.mm(Sigma.mm(V)))
