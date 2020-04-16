#!/usr/bin/env python3
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-04-13 01:00:52
@LastEditTime: 2019-08-09 22:43:15
'''
import argparse
import csv
import json
import logging
import os
import random
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from multiprocessing.dummy import Pool
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import torch
import torchsnooper
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
from scipy import interpolate
from scipy.interpolate import UnivariateSpline as spline
from scipy.stats import truncnorm
from tensorflow_model_optimization.python.core.sparsity.keras import \
    pruning_schedule
from torch.autograd import grad
from torch.utils import data
from torchsummary import summary

# from matrix_parametrization import *
# from rephase2 import *

tf.get_logger().setLevel(logging.ERROR)


tf.logging.propagate = False

##########################
#       torch model      #
##########################


def set_torch_deterministic():
    torch.manual_seed(0)
    np.random.seed(0)
    if(torch.cuda.is_available()):
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(0)


def set_torch_stochastic():
    seed = int(time.time() * 1000) % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if(torch.cuda.is_available()):
        torch.backends.cudnn.deterministic = False
        torch.cuda.manual_seed_all(seed)


def summary_model(model, input):
    summary(model, input)


def save_model(model, path="./checkpoint/model.pt", print_msg=True):
    """Save PyTorch model in path

    Args:
        model (PyTorch model): PyTorch model
        path (str, optional): Full path of PyTorch model. Defaults to "./checkpoint/model.pt".
        print_msg (bool, optional): Control of message print. Defaults to True.
    """
    dir = os.path.dirname(path)
    if(not os.path.exists(dir)):
        os.mkdir(dir)
    try:
        torch.save(model.state_dict(), path)
        if(print_msg):
            print(f"[I] Model saved to {path}")
    except Exception as e:
        if(print_msg):
            print(f"[E] Model failed to be saved to {path}")
        traceback.print_exc(e)


class BestKModelSaver(object):
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self.model_cache = OrderedDict()

    def __insert_model_record(self, acc, dir, checkpoint_name, epoch=None):
        acc = round(acc * 100) / 100
        if(len(self.model_cache) < self.k):
            new_checkpoint_name = f"{checkpoint_name}_acc-{acc:.2f}{'' if epoch is None else '_epoch-'+str(epoch)}"
            path = os.path.join(dir, new_checkpoint_name+".pt")
            self.model_cache[path] = (acc, epoch)
            return path, None
        else:
            min_acc, min_epoch = sorted(list(self.model_cache.values()), key=lambda x: x[0])[0]
            if(acc >= min_acc + 0.01):
                del_checkpoint_name = f"{checkpoint_name}_acc-{min_acc:.2f}{'' if epoch is None else '_epoch-'+str(min_epoch)}"
                del_path = os.path.join(dir, del_checkpoint_name+".pt")
                try:
                    del self.model_cache[del_path]
                except:
                    print(
                        "[W] Cannot remove checkpoint: {} from cache".format(del_path))
                new_checkpoint_name = f"{checkpoint_name}_acc-{acc:.2f}{'' if epoch is None else '_epoch-'+str(epoch)}"
                path = os.path.join(dir, new_checkpoint_name+".pt")
                self.model_cache[path] = (acc, epoch)
                return path, del_path
            elif(acc == min_acc):
                new_checkpoint_name = f"{checkpoint_name}_acc-{acc:.2f}{'' if epoch is None else '_epoch-'+str(epoch)}"
                path = os.path.join(dir, new_checkpoint_name+".pt")
                self.model_cache[path] = (acc, epoch)
                return path, None
            else:
                return None, None

    def save_model(self, model, acc, epoch=None, path="./checkpoint/model.pt", print_msg=True):
        """Save PyTorch model in path

        Args:
            model (PyTorch model): PyTorch model
            path (str, optional): Full path of PyTorch model. Defaults to "./checkpoint/model.pt".
            print_msg (bool, optional): Control of message print. Defaults to True.
        """
        dir = os.path.dirname(path)
        ensure_dir(dir)
        checkpoint_name = os.path.splitext(os.path.basename(path))[0]
        if(isinstance(acc, torch.Tensor)):
            acc = acc.data.item()
        new_path, del_path = self.__insert_model_record(
            acc, dir, checkpoint_name, epoch)

        if(del_path is not None):
            try:
                os.remove(del_path)
                print(f"[I] Model {del_path} is removed")
            except Exception as e:
                if(print_msg):
                    print(f"[E] Model {del_path} failed to be removed")
                traceback.print_exc(e)

        if(new_path is None):
            if(print_msg):
                print(
                    f"[I] Not best {self.k}: {list(reversed(sorted(list(self.model_cache.values()))))}, skip this model ({acc:.2f}): {path}")
        else:
            try:
                torch.save(model.state_dict(), new_path)
                if(print_msg):
                    print(f"[I] Model saved to {new_path}")
            except Exception as e:
                if(print_msg):
                    print(f"[E] Model failed to be saved to {new_path}")
                traceback.print_exc(e)


def load_model(model, path="./checkpoint/model.pt", print_msg=True):
    """Load PyTorch model in path

    Args:
        model (PyTorch model): PyTorch model
        path (str, optional): Full path of PyTorch model. Defaults to "./checkpoint/model.pt".
        print_msg (bool, optional): Control of message print. Defaults to True.
    """
    try:
        state_dict = torch.load(path,  map_location=lambda storage, location: storage)
        cur_state_dict = {key: state_dict[key] for key in model.state_dict()}
        if(len(state_dict) != len(model.state_dict())):
            print(f"[W] Warning! Model is not the same as the checkpoint")

        model.load_state_dict(cur_state_dict)
        if(print_msg):
            print(f"[I] Model loaded from {path}")
    except Exception as e:
        traceback.print_exc(e)
        if(print_msg):
            print(f"[E] Model failed to be loaded from {path}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_converge(trace, epsilon=0.002):
    if(len(trace) <= 1):
        return False
    if(np.abs(trace[-1] - trace[-2]) / (np.abs(trace[-1]) + 1e-8) < epsilon):
        return True
    return False


class ThresholdScheduler(object):
    ''' Intepolation between begin point and end point. step must be within two endpoints
    '''

    def __init__(self, step_beg, step_end, thres_beg, thres_end, mode='tanh'):
        assert mode in {
            "linear", "tanh"}, "Threshold scheduler only supports linear and tanh modes"
        self.mode = mode
        self.step_beg = step_beg
        self.step_end = step_end
        self.thres_beg = thres_beg
        self.thres_end = thres_end
        self.func = self.createFunc()

    def normalize(self, step, factor=2):
        return (step - self.step_beg) / (self.step_end - self.step_beg) * factor

    def createFunc(self):
        if(self.mode == "linear"):
            return lambda x: (self.thres_end - self.thres_beg) * x + self.thres_beg
        elif(self.mode == "tanh"):
            x = self.normalize(
                np.arange(self.step_beg, self.step_end + 1).astype(np.float32))
            y = np.tanh(x) * (self.thres_end - self.thres_beg) + self.thres_beg
            return interpolate.interp1d(x, y)

    def __call__(self, x):
        return self.func(self.normalize(x)).tolist()


class ThresholdScheduler_tf(object):
    ''' smooth increasing threshold with tensorflow model pruning scheduler
    '''

    def __init__(self, step_beg, step_end, thres_beg, thres_end):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.enable_eager_execution(config=config)
        self.step_beg = step_beg
        self.step_end = step_end
        self.thres_beg = thres_beg
        self.thres_end = thres_end
        if(thres_beg < thres_end):
            self.thres_min = thres_beg
            self.thres_range = (thres_end - thres_beg)
            self.descend = False

        else:
            self.thres_min = thres_end
            self.thres_range = (thres_beg - thres_end)
            self.descend = True

        self.pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0, final_sparsity=0.9999999,
            begin_step=self.step_beg, end_step=self.step_end)

    def __call__(self, x):
        if(x < self.step_beg):
            return self.thres_beg
        elif(x > self.step_end):
            return self.thres_end
        res_norm = self.pruning_schedule(x)[1].numpy()
        if(self.descend == False):
            res = res_norm * self.thres_range + self.thres_beg
        else:
            res = self.thres_beg - res_norm * self.thres_range

        if(np.abs(res - self.thres_end) <= 1e-6):
            res = self.thres_end
        return res


class ValueRegister(object):
    def __init__(self, operator, name="", show=True):
        self.op = operator
        self.cache = None
        self.show = show
        self.name = name if len(name) > 0 else "value"

    def register_value(self, x):
        self.cache = self.op(x, self.cache) if self.cache is not None else x
        if(self.show):
            print(f"Recorded {self.name} is {self.cache}")


class ValueTracer(object):
    def __init__(self, show=True):
        self.cache = {}
        self.show = show

    def add_value(self, name, value, step):
        if(name not in self.cache):
            self.cache[name] = {}
        self.cache[name][step] = value
        if(self.show):
            print(f"Recorded {name}: step = {step}, value = {value}")

    def get_trace_by_name(self, name):
        return self.cache.get(name, {})

    def get_all_traces(self):
        return self.cache

    def __len__(self):
        return len(self.cache)

    def get_num_trace(self):
        return len(self.cache)

    def get_len_trace_by_name(self, name):
        return len(self.cache.get(name, {}))

    def dump_trace_to_file(self, name, file):
        if(name not in self.cache):
            print(f"[W] Trace name '{name}' not found in tracer")
            return
        torch.save(self.cache[name], file)
        print(f"[I] Trace {name} saved to {file}")

    def dump_all_traces_to_file(self, file):
        torch.save(self.cache, file)
        print(f"[I] All traces saved to {file}")

    def load_all_traces_from_file(self, file):
        self.cache = torch.load(file)
        return self.cache


class EMA(object):
    def __init__(self, mu):
        super().__init__()
        self.mu = mu
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone().data

    def __call__(self, name, x, mask=None):
        if(name not in self.shadow):
            self.register(name, x)
            return x.data

        old_average = self.shadow[name]
        new_average = (1 - self.mu) * x + self.mu * old_average
        if(mask is not None):
            new_average[mask].copy_(old_average[mask])
        self.shadow[name] = new_average.clone()
        return new_average.data


def export_traces_to_csv(trace_file, csv_file, fieldnames=None):
    traces = torch.load(trace_file)

    with open(csv_file, 'w', newline='') as csvfile:
        if(fieldnames is None):
            fieldnames = list(traces.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        max_len = max([len(traces[field]) for field in fieldnames])

        for idx in range(max_len):
            row = {}
            for field in fieldnames:
                value = traces[field][idx] if idx < len(traces[field]) else ""
                row[field] = value.data.item() if isinstance(
                    value, torch.Tensor) else value
            writer.writerow(row)


def set_learning_rate(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_learning_rate(optimizer):
    return optimizer.param_groups[0]["lr"]

##########################
#         general        #
##########################


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    with open(fname, 'rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with open(fname, 'wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def profile(func=None, timer=True):
    from functools import wraps, partial
    import time
    if (func == None):
        return partial(profile, timer=timer)

    @wraps(func)
    def wrapper(*args, **kw):
        if (timer):
            local_time = time.time()
            res = func(*args, **kw)
            end_time = time.time()
            print('[I] <%s> runtime: %.3f ms' %
                  (func.__name__, (end_time - local_time) * 1000))
        else:
            res = func(*args, **kw)
        return res

    return wrapper


class Timer(object):
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()


class TimerCtx:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


class fullprint:
    'context manager for printing full numpy arrays'

    def __init__(self, **kwargs):
        '''linewidth=75; precision=8'''
        kwargs.setdefault('threshold', np.inf)
        self.opt = kwargs

    def __enter__(self):
        self._opt = np.get_printoptions()
        np.set_printoptions(**self.opt)

    def __exit__(self, type, value, traceback):
        np.set_printoptions(**self._opt)


class Logger(object):
    def __init__(self, console=True, logfile=None, console_level=logging.INFO, logfile_level=logging.INFO):
        super().__init__()
        self.logfile = logfile
        self.console_level = console_level
        self.logifle_level = logfile_level
        assert console == True or logfile is not None, "At least enable one from console or logfile for Logger"
        # 第一步，创建一个logger
        self.logger = logging.getLogger("my_logger")
        self.logger.setLevel(logging.INFO)  # Log等级总开关
        self.logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")

        # 第三步，再创建一个handler，用于输出到控制台
        if(console):
            ch = logging.StreamHandler()
            ch.setLevel(self.console_level)   # 输出到console的log等级的开关
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        if(self.logfile is not None):
            fh = logging.FileHandler(self.logfile, mode='w')
            fh.setLevel(self.logifle_level)   # 输出到file的log等级的开关
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)


class ArgParser(object):
    def __init__(self, file=None):
        super().__init__()
        self.file = file
        self.args = None
        self.parser = argparse.ArgumentParser('Argument Parser')

    # def add_arg(self, option, type, default=None, nargs='+', help=""):
    #     self.parser.add_argument(option, type=type, default=default, help=help)
    def add_arg(self, *args, **keywords):
        self.parser.add_argument(*args, **keywords)

    def parse_args(self):
        self.args = self.parser.parse_args()
        return self.args

    def print_args(self):
        # Print arguments to std out
        # and save argument values to yaml file
        print('Arguments:')
        for p in vars(self.args).items():
            print(f"\t{p[0]:30}{str(p[1]):20}")
        print('\n')

    def dump_args(self):
        if(self.file is None):
            print("[E] Dump failed because the file path is None")
            return
        with open(self.file, 'w') as f:
            yaml.dump(vars(self.args), f, default_flow_style=False)
            print(f"[I] Arguments dumped to {file}")


def print_stat(x):
    if(isinstance(x, torch.Tensor)):
        print(f"min = {x.min().data.item()}, max = {x.max().data.item()}, mean = {x.mean().data.item()}, std = {x.std().data.item()}")
    elif(isinstance(x, np.ndarray)):
        print(f"min = {np.min(x)}, max = {np.max(x)}, mean = {np.mean(x)}, std = {np.std(x)}")


##########################
#       computation      #
##########################

def quant_kaiming_uniform(w, nbit, beta=1.5):
    '''https://arxiv.org/pdf/1802.04680.pdf'''
    if(w.dim() > 2):
        receptive_field = w[0,0,...].numel()
    else:
        receptive_field = 1
    fan_in = w.size(1) * receptive_field
    sigma = 2**(1-nbit)
    L_min = beta * sigma
    L = max(np.sqrt(6/fan_in), L_min)
    return w.clone().uniform_(-L, L)


def quant_kaiming_uniform_(w, nbit, beta=1.5):
    '''https://arxiv.org/pdf/1802.04680.pdf'''
    if(w.dim() > 2):
        receptive_field = w[0,0,...].numel()
    else:
        receptive_field = 1
    fan_in = w.size(1) * receptive_field
    sigma = 2**(1-nbit)
    L = np.sqrt(6/fan_in)
    L_min = beta * sigma
    scale = 2 ** round(np.log2(L_min/L))
    scale = max(scale, 1.0)
    L = max(L, L_min)

    return torch.nn.init.uniform_(w, -L, L), scale


def shift(v, f=1):
    return torch.cat((f * v[[v.size(0) - 1]], v[:-1]))


def Krylov(linear_map, v, n=None):
    if n is None:
        n = v.size(0)
    cols = [v]
    for _ in range(n - 1):
        v = linear_map(v)
        cols.append(v)
    return torch.stack(cols, dim=-1)


def circulant(eigens):
    circ = Krylov(shift, eigens)  # .t()
    return circ


def complex_circulant(eigens):
    circ = Krylov(shift, eigens).transpose(1, 2)  # .t()
    return circ


def complex_mult(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack(
        (X[..., 0] * Y[..., 0] - X[..., 1] * Y[..., 1],
         X[..., 0] * Y[..., 1] + X[..., 1] * Y[..., 0]),
        dim=-1)


def complex_matvec_mult(W, X):
    return torch.sum(complex_mult(W, X.unsqueeze(0).repeat(W.size(0), 1, 1)), dim=1)

def complex_matmul(X, Y):
    assert X.shape[-1] == 2 and Y.shape[-1] == 2, 'Last dimension must be 2'
    return torch.stack([X[..., 0].mm(Y[..., 0]) - X[..., 1].mm(Y[..., 1]), X[...,0].mm(Y[...,1]) + X[..., 1].mm(Y[..., 0])], dim=-1)


def real_to_complex(x):
    return torch.stack((x, torch.zeros_like(x).to(x.device)), dim=-1)


def get_complex_magnitude(x):
    assert x.size(-1) == 2, "[E] Input must be complex Tensor"
    return torch.sqrt(x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1])


def get_complex_energy(x):
    assert x.size(-1) == 2, "[E] Input must be complex Tensor"
    return x[..., 0] * x[..., 0] + x[..., 1] * x[..., 1]


def im2col_2d(W, X=None, stride=1, padding=0):
    if(X is None):
        return W.view(W.size(0), -1), None, None, None
    n_filters, d_filter, h_filter, w_filter = W.size()
    n_x, d_x, h_x, w_x = X.size()

    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    h_out, w_out = int(h_out), int(w_out)
    X_col = torch.nn.functional.unfold(X.view(
        1, -1, h_x, w_x), h_filter, dilation=1, padding=padding, stride=stride).view(n_x, -1, h_out*w_out)
    X_col = X_col.permute(1, 2, 0).contiguous().view(X_col.size(1), -1)
    W_col = W.view(n_filters, -1) # [out_c, in_c*kernel_size*kernel_size]

    return W_col, X_col, h_out, w_out


def check_identity_matrix(W):
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    return (W_numpy.shape[0] == W_numpy.shape[1]) and np.allclose(W_numpy, np.eye(W_numpy.shape[0]))


def check_unitary_matrix(W):
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"
    M = np.dot(W_numpy, W_numpy.T)
    # print(M)
    return check_identity_matrix(M)


def check_equal_tensor(W1, W2):
    if(isinstance(W1, np.ndarray)):
        W1_numpy = W1.copy().astype(np.float64)
    elif(isinstance(W1, torch.Tensor)):
        W1_numpy = W1.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    if(isinstance(W2, np.ndarray)):
        W2_numpy = W2.copy().astype(np.float64)
    elif(isinstance(W2, torch.Tensor)):
        W2_numpy = W2.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"
    return (W1_numpy.shape == W2_numpy.shape) and np.allclose(W1_numpy, W2_numpy)


def merge_chunks(x):
    # x = [H, W, B, B]
    if(isinstance(x, torch.Tensor)):
        h, w, bs = x.size(0), x.size(1), x.size(2)
        x = x.permute(0, 2, 1, 3).contiguous()  # x = [h, bs, w, bs]
        x = x.view(h * bs, w * bs)
    elif(isinstance(x, np.ndarray)):
        h, w, bs = x.shape[0], x.shape[1], x.shape[2]
        x = np.transpose(x, [0, 2, 1, 3])
        x = np.reshape(x, [h * bs, w * bs])
    else:
        raise NotImplementedError
    return x


def partition_chunks(x, bs):
    # x = [H, W]
    if(isinstance(x, torch.Tensor)):
        h, w = x.size(0), x.size(1)
        new_h, new_w = h // bs, w // bs
        x = x.view(new_h, bs, new_w, bs)  # x = (h // bs, bs, w // bs, bs)
        x = x.permute(0, 2, 1, 3).contiguous()  # (h // bs, w // bs, bs, bs)
    elif(isinstance(x, np.ndarray)):
        h, w = x.shape[0], x.shape[1]
        new_h, new_w = h // bs, w // bs
        x = np.reshape(x, [new_h, bs, new_w, bs])
        x = np.transpose(x, [0, 2, 1, 3])
    else:
        raise NotImplementedError

    return x


def gen_boolean_mask_cpu(size, true_prob):
    assert 0 <= true_prob <= 1, f"[E] Wrong probability for True"
    return np.random.choice(a=[False, True], size=size, p=[1-true_prob, true_prob])


def addPhaseDrift(x, std=0.05):
    assert x.size(-1) == 2, "[E] Input of phase drift must be complex Tensor"
    set_torch_stochastic()
    phase_drift = std * \
        torch.randn(x.size()[:-1], dtype=torch.float32).to(x.device)
    set_torch_deterministic()
    phase_drift_complex = torch.stack(
        (torch.cos(phase_drift), torch.sin(phase_drift)), dim=-1)

    return complex_mult(x, phase_drift_complex)


def fftshift_cpu(x, batched=True, dim=None):
    if(isinstance(x, np.ndarray)):
        if(dim is None):
            if(batched):
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.fftshift(x,axes=dim)
    elif(isinstance(x, torch.Tensor)):
        device = x.device
        x = x.cpu().detach().numpy()
        if(dim is None):
            if(batched):
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.fftshift(x,axes=dim)
        out = torch.from_numpy(out).to(device)
    return out

def ifftshift_cpu(x, batched=True, dim=None):
    if(isinstance(x, np.ndarray)):
        if(dim is None):
            if(batched):
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.ifftshift(x,axes=dim)
    elif(isinstance(x, torch.Tensor)):
        device = x.device
        x = x.cpu().detach().numpy()
        if(dim is None):
            if(batched):
                dim = tuple(range(1, len(x.shape)))
            else:
                dim = tuple(range(0, len(x.shape)))
        out = np.fft.ifftshift(x,axes=dim)
        out = torch.from_numpy(out).to(device)
    return out

class uniform_quantize_cpu(object):
    def __init__(self, bits):
        super(uniform_quantize_cpu).__init__()
        self.bits = bits

    def __call__(self, input):
        if self.bits == 32:
            out = input
        elif self.bits == 1:
            out = np.sign(input)
        else:
            n = float(2 ** self.bits - 1)
            out = np.round(input * n) / n
        return out



class phase_quantize_fn_cpu(object):
    def __init__(self, p_bit):
        super(phase_quantize_fn_cpu, self).__init__()
        assert p_bit <= 8 or p_bit == 32
        self.p_bit = p_bit
        self.uniform_q = uniform_quantize_cpu(bits=p_bit)
        self.pi = np.pi

    def __call__(self, x):
        if self.p_bit == 32:
            phase_q = x
        elif self.p_bit == 1:
            E = np.mean(np.abs(x))
            phase_q = self.uniform_q(x / E) * E
        else:
            # phase = torch.tanh(x)
            # phase = phase / 2 / torch.max(torch.abs(phase)) + 0.5
            phase = x / 2 / self.pi + 0.5
            # phase_q = 2 * self.uniform_q(phase) - 1
            phase_q = self.uniform_q(phase) * 2 * self.pi - self.pi
        return phase_q


class voltage_quantize_fn_cpu(object):
    def __init__(self, v_bit, v_pi, v_max):
        super(voltage_quantize_fn_cpu, self).__init__()
        assert 0 < v_bit <= 32
        self.v_bit = v_bit
        self.v_pi = v_pi
        self.v_max = v_max
        self.gamma = np.pi / (self.v_pi**2)
        self.uniform_q = uniform_quantize_cpu(bits=v_bit)
        self.pi = np.pi

    def __call__(self, x, voltage_mask_old=None, voltage_mask_new=None, voltage_backup=None, strict_mask=True):
        if self.v_bit == 32:
            voltage_q = x
        elif self.v_bit == 1:
            E = np.mean(np.abs(x))
            voltage_q = self.uniform_q(x / E) * E
        else:
            min_V = 0
            ### max voltage is determined by the voltage supply, not the phase shifter's characteristics!!! ###
            # max_V = np.sqrt(2*self.pi/self.gamma)
            max_V = self.v_max
            voltage = (x - min_V) / (max_V - min_V)
            # phase_q = 2 * self.uniform_q(phase) - 1
            voltage_q = self.uniform_q(voltage) * (max_V - min_V) + min_V

            if(voltage_mask_old is not None and voltage_mask_new is not None and voltage_backup is not None):
                if(strict_mask == True):
                    # strict mask will always fix masked voltages, even though they are not covered in the new mask
                    # "1" in mask indicates to apply quantization
                    voltage_mask_newly_marked = voltage_mask_new ^ voltage_mask_old
                    voltage_q_tmp = x.copy()
                    # maintain voltages that have already been masked
                    voltage_q_tmp[voltage_mask_old] = voltage_backup[voltage_mask_old]
                    # quantize new voltages those are marked in the new mask
                    # print("any newly marked voltages:", voltage_mask_newly_marked.any())
                    voltage_q_tmp[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]
                    # only update newly quantized voltages, previously quantized voltages are maintained
                    # if (voltage_backup[voltage_mask_newly_marked].sum() > 1e-4):
                    #     print(voltage_backup[voltage_mask_newly_marked])

                    voltage_backup[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]

                    voltage_q = voltage_q_tmp
                else:
                    # non-strict mask will make unmasked voltages trainable again
                    voltage_q_tmp = x.copy()
                    voltage_mask_old = voltage_mask_old & voltage_mask_new
                    voltage_mask_newly_marked = (
                        ~voltage_mask_old) & voltage_mask_new
                    # maintain voltages that have already been masked and being masked in the new mask
                    voltage_q_tmp[voltage_mask_old] = voltage_backup[voltage_mask_old]
                    # quantize new voltages those are marked in the new mask
                    voltage_q_tmp[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]

                    voltage_backup[voltage_mask_newly_marked] = voltage_q[voltage_mask_newly_marked]
                    voltage_q = voltage_q_tmp

        return voltage_q


def clip_to_valid_quantized_voltage_cpu(voltages, gamma, v_bit, v_max, wrap_around=False):
    v_2pi = np.sqrt(2 * np.pi / gamma)
    v_interval = v_max / (2**v_bit-1)
    if(wrap_around):
        mask = voltages >= v_2pi
        voltages[mask] = 0
        # invalid_voltages = voltages[mask]
        # invalid_phases = gamma * invalid_voltages * invalid_voltages
        # invalid_phases -= 2 * np.pi
        # valid_voltages = np.sqrt(invalid_phases / gamma)
        # valid_voltages_q = np.round(valid_voltages / v_interval) * v_interval
        # voltages[mask] = valid_voltages_q
    else:
        voltages[voltages > v_2pi] -= v_interval
    return voltages


def voltage_to_phase_cpu(voltages, gamma):
    # phases = -np.clip(gamma * voltages * voltages, a_min=0, a_max=2 * np.pi)
    # change phase range from [-2pi,0] to [-pi,pi]
    phases = -gamma * voltages * voltages
    pi_2 = 2*np.pi
    phases[phases <= -pi_2] += pi_2
    phases[phases < -np.pi] += pi_2
    return phases


def phase_to_voltage_cpu(phases, gamma):
    pi = np.pi
    phases_tmp = phases.copy()
    phases_tmp[phases_tmp > 0] -= 2 * pi  # change phase lead to phase lag
    voltage_max = np.sqrt((2 * pi) / gamma)
    voltages = np.clip(np.sqrt(np.abs(phases_tmp / gamma)),
                       a_min=0, a_max=voltage_max)
    return voltages


def quantize_phase_of_matrix_cpu(W, p_bit, output_device=torch.device("cuda")):
    assert isinstance(
        p_bit, int) and p_bit >= 1, "[E] quantization bit must be integer larger than 1"

    decomposer = RealUnitaryDecomposer()
    phase_quantize_fn = phase_quantize_fn_cpu(p_bit=p_bit)

    if(isinstance(W, np.ndarray)):
        W_numpy = W.astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    M, N = W_numpy.shape[0], W_numpy.shape[1]
    U, Sigma, V = np.linalg.svd(W_numpy, full_matrices=True)

    Sigma = np.diag(Sigma)
    if(M > N):
        Sigma = np.concatenate([Sigma, np.zeros([M - N, N])], axis=0)
    elif(M < N):
        Sigma = np.concatenate([Sigma, np.zeros([M, N - M])], axis=1)

    delta_list_U, phi_mat_U = decomposer.decompose(U)
    delta_list_V, phi_mat_V = decomposer.decompose(V)

    phi_list_U = np.zeros([M*(M-1)//2], dtype=np.float64)
    phi_list_V = np.zeros([N*(N-1)//2], dtype=np.float64)
    count = 0
    for i in range(M):
        for j in range(M - i - 1):
            phi_list_U[count] = phi_mat_U[i, j]
            count += 1
    count = 0
    for i in range(N):
        for j in range(N - i - 1):
            phi_list_V[count] = phi_mat_V[i, j]
            count += 1

    phi_list_U_q = phase_quantize_fn(phi_list_U)
    phi_list_V_q = phase_quantize_fn(phi_list_V)
    # phi_list_U_q = phi_list_U
    # phi_list_V_q = phi_list_V

    count = 0
    for i in range(M):
        for j in range(M - i - 1):
            phi_mat_U[i, j] = phi_list_U_q[count]
            count += 1
    count = 0
    for i in range(N):
        for j in range(N - i - 1):
            phi_mat_V[i, j] = phi_list_V_q[count]
            count += 1

    U_recon = decomposer.reconstruct_2(delta_list_U, phi_mat_U)
    V_recon = decomposer.reconstruct_2(delta_list_V, phi_mat_V)

    U_recon = torch.from_numpy(U_recon).to(output_device)
    V_recon = torch.from_numpy(V_recon).to(output_device)
    Sigma = torch.from_numpy(Sigma).to(output_device)

    W_recon = torch.mm(U_recon, torch.mm(Sigma, V_recon)).to(torch.float32)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(W_numpy.reshape([-1]), bins=256, range=[-1, 1])
    # plt.subplot(2,1,2)
    # plt.hist(W_recon.detach().cpu().numpy().reshape([-1]), bins=256, range=[-1, 1])
    # plt.show()
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(phi_list_U.reshape([-1]), bins=256, range=[-np.pi, np.pi])
    # plt.subplot(2,1,2)
    # plt.hist(phi_list_U_q.reshape([-1]), bins=256, range=[-np.pi, np.pi])
    # plt.show()
    return W_recon


def upper_triangle_to_vector_cpu(mat):
    count = 0
    shape = mat.shape
    if(len(shape) == 2):
        N = shape[0]
        vector = np.zeros(shape=[N*(N-1)//2], dtype=mat.dtype)
        for i in range(N):
            for j in range(N - i - 1):
                vector[count] = mat[i, j]
                count += 1
    else:
        N = shape[-1]
        vector = np.zeros(shape=list(
            list(shape[:-2])+[N*(N-1)//2]), dtype=mat.dtype)
        for i in range(N):
            for j in range(N - i - 1):
                vector[..., count] = mat[..., i, j]
                count += 1

    return vector


def vector_to_upper_triangle_cpu(vec):
    count = 0
    shape = vec.shape
    if(len(shape) == 1):
        M = vec.shape[0]
        N = (1 + int(np.sqrt(1 + 8 * M))) // 2
        mat = np.zeros(shape=[N, N], dtype=vec.dtype)
        for i in range(N):
            for j in range(N - i - 1):
                mat[i, j] = vec[count]
                count += 1
    else:
        M = shape[-1]
        N = (1 + int(np.sqrt(1 + 8 * M))) // 2
        mat = np.zeros(shape=list(shape[:-1]) + [N, N], dtype=vec.dtype)
        for i in range(N):
            for j in range(N - i - 1):
                mat[..., i, j] = vec[..., count]
                count += 1
    return mat


def projection_matrix_to_unitary_cpu(W):
    U, S, V = np.linalg.svd(W, full_matrices=True)
    U_refine = np.matmul(U, V)
    return U_refine


def real_matrix_parametrization_cpu(W):
    decomposer = RealUnitaryDecomposer()
    M, N = W.shape[0], W.shape[1]
    U, Sigma, V = np.linalg.svd(W, full_matrices=True)

    Sigma = np.diag(Sigma)
    if(M > N):
        Sigma = np.concatenate([Sigma, np.zeros([M - N, N])], axis=0)
    elif(M < N):
        Sigma = np.concatenate([Sigma, np.zeros([M, N - M])], axis=1)

    delta_list_U, phi_mat_U = decomposer.decompose(U)
    delta_list_V, phi_mat_V = decomposer.decompose(V)

    phi_list_U = upper_triangle_to_vector_cpu(phi_mat_U)
    phi_list_V = upper_triangle_to_vector_cpu(phi_mat_V)

    return Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V


def real_matrix_parametrization_cpu_ref(W):
    M, N = W.shape[0], W.shape[1]
    U, Sigma, V = np.linalg.svd(W, full_matrices=True)

    Sigma = np.diag(Sigma)
    if(M > N):
        Sigma = np.concatenate([Sigma, np.zeros([M - N, N])], axis=0)
    elif(M < N):
        Sigma = np.concatenate([Sigma, np.zeros([M, N - M])], axis=1)

    delta_list_U, phi_mat_U = decompose_ref(U)
    delta_list_V, phi_mat_V = decompose_ref(V)

    phi_list_U = upper_triangle_to_vector_cpu(phi_mat_U)
    phi_list_V = upper_triangle_to_vector_cpu(phi_mat_V)

    return Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V


def real_matrix_reconstruction_cpu(Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V):
    decomposer = RealUnitaryDecomposer()
    phi_mat_U = vector_to_upper_triangle_cpu(phi_list_U)
    phi_mat_V = vector_to_upper_triangle_cpu(phi_list_V)

    U_recon = decomposer.reconstruct_2(delta_list_U, phi_mat_U)
    V_recon = decomposer.reconstruct_2(delta_list_V, phi_mat_V)
    # print("checkU:",decomposer.checkUnitary(U_recon), decomposer.checkUnitary(V_recon))

    W_recon = np.dot(U_recon, np.dot(Sigma, V_recon))

    return W_recon


def apply_weight_decay(W, decay_rate, learning_rate, mask=None):
    # in mask, 1 represents fixed variables, 0 represents trainable variables
    if(mask is not None):
        W[~mask] -= W[~mask] * decay_rate * learning_rate
    else:
        W -= W * decay_rate * learning_rate


def quantize_voltage_of_matrix_cpu(W, v_bit, v_pi=4.36, v_max=10.8, voltage_mask_U=None, voltage_backup_U=None, voltage_mask_V=None, voltage_backup_V=None, quantize_voltage_percentile=0, clamp_small_phase_lead_percentile=1, output_device=torch.device("cuda")):
    assert isinstance(
        v_bit, int) and v_bit >= 1, "[E] quantization bit must be integer larger than 1"
    assert 0 < clamp_small_phase_lead_percentile <= 1, "[E] Clamp phase lead percentile must be within (0, 1]"
    assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0, 1]"

    gamma = np.pi / (v_pi**2)
    voltage_quantize_fn = voltage_quantize_fn_cpu(
        v_bit=v_bit, v_pi=v_pi, v_max=v_max)

    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V = real_matrix_parametrization_cpu(
        W_numpy)

    v_list_U = phase_to_voltage_cpu(phi_list_U, gamma)
    v_list_V = phase_to_voltage_cpu(phi_list_V, gamma)

    # if(clamp_small_phase_lead_percentile < 1):
    #     thres = np.percentile(v_list_U, clamp_small_phase_lead_percentile*100)
    #     v_list_U[v_list_U > thres] = 0
    #     thres = np.percentile(v_list_V, clamp_small_phase_lead_percentile*100)
    #     v_list_V[v_list_V > thres] = 0
    if(voltage_mask_U is not None and voltage_backup_U is not None and voltage_mask_V is not None and voltage_backup_V is not None):
        v_thres_U = np.percentile(
            v_list_U, (1-quantize_voltage_percentile)*100)
        voltage_mask_U_old = voltage_mask_U.copy()
        voltage_mask_U_new = voltage_mask_U_old | (v_list_U >= v_thres_U)
        v_thres_V = np.percentile(
            v_list_V, (1-quantize_voltage_percentile)*100)
        voltage_mask_V_old = voltage_mask_V.copy()
        voltage_mask_V_new = voltage_mask_V_old | (v_list_V >= v_thres_V)
        print(f"[I] Voltage_threshold: U={v_thres_U}, V={v_thres_V}")

        v_list_U_q = voltage_quantize_fn(
            v_list_U, voltage_mask_U_old, voltage_mask_U_new, voltage_backup_U)
        v_list_V_q = voltage_quantize_fn(
            v_list_V, voltage_mask_V_old, voltage_mask_V_new, voltage_backup_V)
        voltage_mask_U[:] = voltage_mask_U_new[:]
        voltage_mask_V[:] = voltage_mask_V_new[:]
        print(f"{voltage_backup_U.max(), voltage_backup_U.min(),voltage_backup_V.max(), voltage_backup_V.min()}")
        print(voltage_mask_U.all(), voltage_mask_V.all())
        if(voltage_mask_U.all() and voltage_mask_V.all()):
            print("mask is all True")
            print("sigma", Sigma)
            print("delta:", delta_list_U)
            print("v_list_U_q:", v_list_U_q)
            print("v_backup_U", voltage_backup_U)
    else:
        v_list_U_q = voltage_quantize_fn(v_list_U, None, None, None)
        v_list_V_q = voltage_quantize_fn(v_list_V, None, None, None)
        # print("before Q:", v_list_U)
        # print("after Q:", v_list_U_q)
        # print("backup:", voltage_backup_U)
        print("Mask Is All True")
        print("sigma", Sigma)
        print("delta:", delta_list_U)
        print("v_list_U_q:", v_list_U_q)
        print("v_backup_U", voltage_backup_U)

    # print("v_list_U_q:", v_list_U_q)
    # print("v_backup_U:", voltage_backup_U)
    # print("v_list_V_q:", v_list_V_q)
    # print("v_backup_V:", voltage_backup_V)

    phi_list_U_q = voltage_to_phase_cpu(v_list_U_q, gamma)
    phi_list_V_q = voltage_to_phase_cpu(v_list_V_q, gamma)
    # phi_list_U_q = phi_list_U
    # phi_list_V_q = phi_list_V

    W_recon = real_matrix_reconstruction_cpu(
        Sigma, delta_list_U, phi_list_U_q, delta_list_V, phi_list_V_q)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

    res = check_equal_tensor(W, W_recon)
    print("[I] checkEqual: ", res)
    # print("S:", Sigma)
    # print("delta_U:", delta_list_U)
    # print("delta_V:", delta_list_V)
    print("W:", W)
    print("W_rec:", W_recon)

    # bin=2**v_bit
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(W_numpy.reshape([-1]), bins=bin, range=[-1, 1])
    # plt.subplot(2,1,2)
    # plt.hist(W_recon.detach().cpu().numpy().reshape([-1]), bins=bin, range=[-1, 1])

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(phi_list_U.reshape([-1]), bins=bin, range=[-np.pi, np.pi])
    # plt.subplot(2,1,2)
    # plt.hist(phi_list_U_q.reshape([-1]), bins=bin, range=[-np.pi, np.pi])

    # v_max = np.sqrt(2*np.pi/gamma)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(v_list_U.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.subplot(2,1,2)
    # plt.hist(v_list_U_q.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.show()

    return W_recon


def maintain_quantized_voltage_cpu(W, v_pi, voltage_mask_U, voltage_backup_U, voltage_mask_V, voltage_backup_V, output_device=torch.device("cuda")):
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V = real_matrix_parametrization_cpu(
        W_numpy)

    gamma = np.pi / (v_pi**2)
    v_list_U = phase_to_voltage_cpu(phi_list_U, gamma)
    v_list_V = phase_to_voltage_cpu(phi_list_V, gamma)
    # print("maintain:",voltage_mask_U.all(), voltage_mask_V.all())
    v_list_U[voltage_mask_U] = voltage_backup_U[voltage_mask_U]
    v_list_V[voltage_mask_V] = voltage_backup_V[voltage_mask_V]

    phi_list_U = voltage_to_phase_cpu(v_list_U, gamma)
    phi_list_V = voltage_to_phase_cpu(v_list_V, gamma)

    W_recon = real_matrix_reconstruction_cpu(
        Sigma, delta_list_U, phi_list_U, delta_list_V, phi_list_V)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)
    W.data.copy_(W_recon)


def quantize_voltage_of_unitary_cpu(W, v_bit, v_pi=4.36, v_max=10.8, voltage_mask=None, voltage_backup=None, quantize_voltage_percentile=0, strict_mask=True, clamp_small_phase_lead_percentile=1, output_device=torch.device("cuda")):
    assert isinstance(
        v_bit, int) and v_bit >= 1, "[E] quantization bit must be integer larger than 1"
    assert 0 < clamp_small_phase_lead_percentile <= 1, "[E] Clamp phase lead percentile must be within (0, 1]"
    assert 0 <= quantize_voltage_percentile <= 1, "[E] Quantize voltage percentile must be within [0, 1]"

    gamma = np.pi / (v_pi**2)
    voltage_quantize_fn = voltage_quantize_fn_cpu(
        v_bit=v_bit, v_pi=v_pi, v_max=v_max)

    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"
    batch_mode = len(W_numpy.shape) > 2

    if(batch_mode):
        decomposer = RealUnitaryDecomposerBatch()
        delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
    else:
        decomposer = RealUnitaryDecomposer()
        delta_list, phi_mat = decomposer.decompose(W_numpy)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)

    v_list = phase_to_voltage_cpu(phi_list, gamma)

    if(voltage_mask is not None and voltage_backup is not None):
        v_thres = np.percentile(
            v_list, (1-quantize_voltage_percentile)*100)
        voltage_mask_old = voltage_mask.copy()
        if(strict_mask == True):
            voltage_mask_new = voltage_mask_old | (v_list >= v_thres)
        else:
            voltage_mask_new = v_list >= v_thres
        # print(f"[I] Voltage_threshold: U={v_thres}")

        v_list_q = voltage_quantize_fn(
            v_list, voltage_mask_old, voltage_mask_new, voltage_backup, strict_mask)

        voltage_mask[:] = voltage_mask_new[:]

        # print(f"{voltage_backup.max(), voltage_backup.min()}")
        # if(voltage_mask.all()):
        #     print("mask is all True")
        #     print("delta:",delta_list)
        #     print("v_list_q:",v_list_q)
        #     print("v_backup",voltage_backup)
    else:
        v_list_q = voltage_quantize_fn(v_list, None, None, None, True)
        v_list_q = clip_to_valid_quantized_voltage_cpu(
            v_list_q, gamma, v_bit, v_max, wrap_around=True)
        # print("before Q:", v_list_U)
        # print("after Q:", v_list_U_q)
        # print("backup:", voltage_backup_U)
        # print("Mask Is All True")
        # print("delta:",delta_list)
        # print("v_list_q:",v_list_q)
        # print("v_backup",voltage_backup)

    # print("v_list_U_q:", v_list_U_q)
    # print("v_backup_U:", voltage_backup_U)
    # print("v_list_V_q:", v_list_V_q)
    # print("v_backup_V:", voltage_backup_V)

    phi_list_q = voltage_to_phase_cpu(v_list_q, gamma)
    # phi_list_U_q = phi_list_U
    # phi_list_V_q = phi_list_V
    phi_mat_q = vector_to_upper_triangle_cpu(phi_list_q)

    if(batch_mode):
        W_recon = decomposer.reconstruct_2_batch(delta_list, phi_mat_q)
    else:
        W_recon = decomposer.reconstruct_2(delta_list, phi_mat_q)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

    # res = check_equal_tensor(W, W_recon)
    # print("[I] checkEqual: ", res)
    # print("S:", Sigma)
    # print("delta_U:", delta_list_U)
    # print("delta_V:", delta_list_V)
    # print("W:", W)
    # print("W_rec:", W_recon)

    # bin=2**v_bit
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(W_numpy.reshape([-1]), bins=bin, range=[-1, 1])
    # plt.subplot(2,1,2)
    # plt.hist(W_recon.detach().cpu().numpy().reshape([-1]), bins=bin, range=[-1, 1])

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(phi_list_U.reshape([-1]), bins=bin, range=[-np.pi, np.pi])
    # plt.subplot(2,1,2)
    # plt.hist(phi_list_U_q.reshape([-1]), bins=bin, range=[-np.pi, np.pi])

    # v_max = np.sqrt(2*np.pi/gamma)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(v_list_U.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.subplot(2,1,2)
    # plt.hist(v_list_U_q.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.show()

    return W_recon


def maintain_quantized_voltage_of_unitary_cpu(W, v_pi, voltage_mask, voltage_backup, gamma_noise_std=0, weight_decay_rate=0, learning_rate=0, clip_voltage=False, lower_thres=float("-inf"), upper_thres=float("inf"), output_device=torch.device("cuda")):
    assert gamma_noise_std >= 0, "[E] Gamma noise standard deviation must be non-negative"
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    gamma = np.pi / (v_pi**2)

    batch_mode = len(W_numpy.shape) > 2
    if(batch_mode):
        decomposer = RealUnitaryDecomposerBatch()
        delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
    else:
        decomposer = RealUnitaryDecomposer()
        delta_list, phi_mat = decomposer.decompose(W_numpy)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)

    v_list = phase_to_voltage_cpu(phi_list, gamma)

    v_list[voltage_mask] = voltage_backup[voltage_mask]

    if(weight_decay_rate > 1e-6 and learning_rate > 1e-6):
        apply_weight_decay(v_list, decay_rate=weight_decay_rate,
                           learning_rate=learning_rate, mask=voltage_mask)

    if(clip_voltage == True):
        v_max = np.sqrt(2*np.pi/gamma)
        lower_mask_1 = (v_list > (lower_thres + 0)/2) & (v_list < lower_thres)
        lower_mask_2 = v_list <= (lower_thres + 0)/2
        upper_mask_1 = (v_list > upper_thres) & (
            v_list < (upper_thres + v_max)/2)
        upper_mask_2 = (v_list >= (upper_thres + v_max)/2)
        v_list[lower_mask_1] = lower_thres
        v_list[lower_mask_2] = 0
        v_list[upper_mask_1] = upper_thres
        v_list[upper_mask_2] = 0

    if(gamma_noise_std > 1e-4):
        pool = Pool(2)
        # reconstruct unitary matrix without noise
        phi_list = voltage_to_phase_cpu(v_list, gamma)
        phi_mat = vector_to_upper_triangle_cpu(phi_list)
        # W_recon = decomposer.reconstruct_2(delta_list, phi_mat)
        # W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

        # reconstruct unitary matrix with gamma noise
        N = W_numpy.shape[0]
        gamma_with_noise = add_gaussian_noise_cpu(np.zeros_like(
            v_list, dtype=np.float64), noise_mean=gamma, noise_std=gamma_noise_std, trunc_range=())
        # Must add noise to all voltages to model the correct noise distribution
        # gamma_with_noise[~voltage_mask] = gamma ### only add noise to masked voltages, avoid aggressive noise error?
        phi_list_n = voltage_to_phase_cpu(v_list, gamma_with_noise)
        phi_mat_n = vector_to_upper_triangle_cpu(phi_list_n)
        # W_recon_n = decomposer.reconstruct_2(delta_list, phi_mat_n)
        # W_recon_n = torch.from_numpy(W_recon_n).to(torch.float32).to(output_device)
        if(batch_mode):
            W_recon, W_recon_n = pool.map(lambda x: torch.from_numpy(decomposer.reconstruct_2_batch(
                delta_list=delta_list, phi_mat=x)).to(torch.float32).to(output_device), [phi_mat, phi_mat_n])
        else:
            W_recon, W_recon_n = pool.map(lambda x: torch.from_numpy(decomposer.reconstruct_2(
                delta_list=delta_list, phi_mat=x)).to(torch.float32).to(output_device), [phi_mat, phi_mat_n])
        pool.close()

        return W_recon, W_recon_n

    else:
        phi_list = voltage_to_phase_cpu(v_list, gamma)
        phi_mat = vector_to_upper_triangle_cpu(phi_list)
        if(batch_mode):
            W_recon = decomposer.reconstruct_2_batch(delta_list, phi_mat)
        else:
            W_recon = decomposer.reconstruct_2(delta_list, phi_mat)
        W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

        return W_recon, None
    # W.data.copy_(W_recon)


def clip_voltage_of_unitary_cpu(W, v_pi, lower_thres=0, upper_thres=float('inf'), voltage_mask=None, voltage_backup=None, output_device=torch.device("cuda")):
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    gamma = np.pi / (v_pi**2)

    N = W_numpy.shape[0]

    decomposer = RealUnitaryDecomposer()
    delta_list, phi_mat = decomposer.decompose(W_numpy)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)

    v_list = phase_to_voltage_cpu(phi_list, gamma)

    if(voltage_mask is None and voltage_backup is None):
        v_max = np.sqrt(2*np.pi/gamma)
        # lower_thres = np.percentile(v_list, lower_perc * 100)
        # upper_thres = np.percentile(v_list, upper_perc * 100)
        lower_mask = v_list < lower_thres
        upper_mask_1 = (v_list > upper_thres) & (
            v_list < (upper_thres + v_max)/2)
        upper_mask_2 = (v_list >= (upper_thres + v_max)/2)
        v_list[lower_mask] = lower_thres
        v_list[upper_mask_1] = upper_thres
        v_list[upper_mask_2] = 0
    else:
        raise NotImplementedError

    phi_list = voltage_to_phase_cpu(v_list, gamma)

    phi_mat = vector_to_upper_triangle_cpu(phi_list)

    W_recon = decomposer.reconstruct_2(delta_list, phi_mat)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

    return W_recon


def conditional_update_voltage_of_unitary_cpu(W, v_bit, v_pi, v_max, lambda3, voltage_mask, voltage_backup, gamma_noise_std=0, weight_decay_rate=0, learning_rate=0, clip_voltage=False, lower_thres=float("-inf"), upper_thres=float("inf"), return_ori=True, output_device=torch.device("cuda")):
    assert gamma_noise_std >= 0, "[E] Gamma noise standard deviation must be non-negative"
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    gamma = np.pi / (v_pi**2)
    voltage_quantize_fn = voltage_quantize_fn_cpu(
        v_bit=v_bit, v_pi=v_pi, v_max=v_max)

    batch_mode = len(W_numpy.shape) > 2
    if(batch_mode):
        decomposer = RealUnitaryDecomposerBatch()
        delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
    else:
        decomposer = RealUnitaryDecomposer()
        delta_list, phi_mat = decomposer.decompose(W_numpy)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)
    v_list = phase_to_voltage_cpu(phi_list, gamma)

    if(weight_decay_rate > 1e-6 and learning_rate > 1e-6):
        apply_weight_decay(v_list, decay_rate=weight_decay_rate,
                           learning_rate=learning_rate, mask=voltage_mask)

    v_list_q = voltage_quantize_fn(v_list, None, None, None, True)
    v_list_q = clip_to_valid_quantized_voltage_cpu(
        v_list_q, gamma, v_bit, v_max, wrap_around=True)

    # v_list = (1 - lambda3) * v_list + lambda3 * v_list_q # conditional does not work here!!!
    v_list = v_list_q

    if(clip_voltage == True):
        v_max = np.sqrt(2*np.pi/gamma)
        # lower_mask_1 = (v_list_q > (lower_thres + 0)/2) & (v_list_q < lower_thres)
        # lower_mask_2 = v_list_q <= (lower_thres + 0)/2
        upper_mask_1 = (v_list > upper_thres) & (
            v_list < (upper_thres + v_max)/2)
        upper_mask_2 = v_list >= (upper_thres + v_max)/2
        # v_list_q[lower_mask_1] = lower_thres
        # v_list_q[lower_mask_2] = 0
        v_list[upper_mask_1] = upper_thres
        v_list[upper_mask_2] = 0
        # v_list_q[v_list_q > upper_thres] = upper_thres

    if(gamma_noise_std > 1e-4):
        pool = Pool(2)
        # reconstruct unitary matrix without noise

        # W_recon = decomposer.reconstruct_2(delta_list, phi_mat)
        # W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

        # reconstruct unitary matrix with gamma noise
        N = W_numpy.shape[0]
        gamma_with_noise = add_gaussian_noise_cpu(np.zeros_like(
            v_list, dtype=np.float64), noise_mean=gamma, noise_std=gamma_noise_std, trunc_range=())
        # Must add noise to all voltages to model the correct noise distribution
        # gamma_with_noise[~voltage_mask] = gamma ### only add noise to masked voltages, avoid aggressive noise error?
        phi_list_n = voltage_to_phase_cpu(v_list, gamma_with_noise)
        phi_mat_n = vector_to_upper_triangle_cpu(phi_list_n)
        # W_recon_n = decomposer.reconstruct_2(delta_list, phi_mat_n)
        # W_recon_n = torch.from_numpy(W_recon_n).to(torch.float32).to(output_device)
        if(batch_mode):
            if(return_ori):
                phi_list = voltage_to_phase_cpu(v_list, gamma)
                phi_mat = vector_to_upper_triangle_cpu(phi_list)
                W_recon, W_recon_n = pool.map(lambda x: torch.from_numpy(decomposer.reconstruct_2_batch(
                    delta_list=delta_list, phi_mat=x)).to(torch.float32).to(output_device), [phi_mat, phi_mat_n])
            else:
                W_recon = None
                W_recon_n = torch.from_numpy(decomposer.reconstruct_2_batch(
                    delta_list=delta_list, phi_mat=phi_mat_n)).to(torch.float32).to(output_device)
        else:
            if(return_ori):
                phi_list = voltage_to_phase_cpu(v_list, gamma)
                phi_mat = vector_to_upper_triangle_cpu(phi_list)
                W_recon, W_recon_n = pool.map(lambda x: torch.from_numpy(decomposer.reconstruct_2(
                    delta_list=delta_list, phi_mat=x)).to(torch.float32).to(output_device), [phi_mat, phi_mat_n])
            else:
                W_recon = None
                W_recon_n = torch.from_numpy(decomposer.reconstruct_2(
                    delta_list=delta_list, phi_mat=phi_mat_n)).to(torch.float32).to(output_device)
        pool.close()

        return W_recon, W_recon_n

    else:
        # reconstruct unitary matrix without noise
        phi_list = voltage_to_phase_cpu(v_list, gamma)
        phi_mat = vector_to_upper_triangle_cpu(phi_list)

        if(batch_mode):
            W_recon = torch.from_numpy(decomposer.reconstruct_2_batch(
                delta_list=delta_list, phi_mat=phi_mat)).to(torch.float32).to(output_device)
            W_recon_n = W_recon.clone()
        else:
            W_recon = torch.from_numpy(decomposer.reconstruct_2(
                delta_list=delta_list, phi_mat=phi_mat)).to(torch.float32).to(output_device)
            W_recon_n = W_recon.clone()

        return W_recon, W_recon_n
    # W.data.copy_(W_recon)


def add_gaussian_noise_cpu(W, noise_mean=0, noise_std=0.002, trunc_range=()):
    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"
    if(not trunc_range):
        noises = np.random.normal(noise_mean, noise_std, W_numpy.shape)
    else:
        a = (trunc_range[0] - noise_mean) / noise_std
        b = (trunc_range[1] - noise_mean) / noise_std
        noises = truncnorm.rvs(
            a, b, loc=noise_mean, scale=noise_std, size=W_numpy.shape, random_state=None)
    return W_numpy + noises


def add_gamma_noise_to_unitary_cpu(W, v_pi=4.36, gamma_noise_std=0.002, output_device=torch.device("cuda")):
    assert 0 <= gamma_noise_std <= 1, "[E] Gamma noise standard diviation must be within [0, 1]"

    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    gamma = np.pi / (v_pi**2)

    batch_mode = len(W_numpy.shape) > 2
    if(batch_mode):
        decomposer = RealUnitaryDecomposerBatch()
        delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
    else:
        decomposer = RealUnitaryDecomposer()
        delta_list, phi_mat = decomposer.decompose(W_numpy)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)

    v_list = phase_to_voltage_cpu(phi_list, gamma)

    gamma_with_noise = add_gaussian_noise_cpu(np.zeros_like(
        v_list, dtype=np.float64), noise_mean=gamma, noise_std=gamma_noise_std, trunc_range=())

    # thres = np.percentile(v_list, 10)
    # thres2 = np.percentile(v_list, 0)
    # mask = (v_list > thres) ^ (v_list > thres2)
    # gamma_with_noise[~mask] = gamma

    phi_list_n = voltage_to_phase_cpu(v_list, gamma_with_noise)
    # phi_list_U_q = phi_list_U
    # phi_list_V_q = phi_list_V
    phi_mat_n = vector_to_upper_triangle_cpu(phi_list_n)
    if(batch_mode):
        W_recon = decomposer.reconstruct_2_batch(delta_list, phi_mat_n)
    else:
        W_recon = decomposer.reconstruct_2(delta_list, phi_mat_n)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

    # res = check_equal_tensor(W, W_recon)
    # print("[I] checkEqual: ", res)
    # print("S:", Sigma)
    # print("delta_U:", delta_list_U)
    # print("delta_V:", delta_list_V)
    # print("W:", W)
    # print("W_rec:", W_recon)

    # bin=2**v_bit
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(W_numpy.reshape([-1]), bins=bin, range=[-1, 1])
    # plt.subplot(2,1,2)
    # plt.hist(W_recon.detach().cpu().numpy().reshape([-1]), bins=bin, range=[-1, 1])

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(phi_list_U.reshape([-1]), bins=bin, range=[-np.pi, np.pi])
    # plt.subplot(2,1,2)
    # plt.hist(phi_list_U_q.reshape([-1]), bins=bin, range=[-np.pi, np.pi])

    # v_max = np.sqrt(2*np.pi/gamma)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(v_list_U.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.subplot(2,1,2)
    # plt.hist(v_list_U_q.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.show()

    return W_recon


def add_phase_noise_to_unitary_cpu(W, phase_noise_std=0.002, protect_mask=None, output_device=torch.device("cuda")):
    assert 0 <= phase_noise_std <= 1, "[E] Phase noise standard diviation must be within [0, 1]"

    if(isinstance(W, np.ndarray)):
        W_numpy = W.copy().astype(np.float64)
    elif(isinstance(W, torch.Tensor)):
        W_numpy = W.detach().cpu().numpy().copy().astype(np.float64)
    else:
        assert 0, "[E] Array type not supported, must be numpy.ndarray or torch.Tensor"

    batch_mode = len(W_numpy.shape) > 2
    if(batch_mode):
        decomposer = RealUnitaryDecomposerBatch()
        delta_list, phi_mat = decomposer.decompose_batch(W_numpy)
    else:
        decomposer = RealUnitaryDecomposer()
        delta_list, phi_mat = decomposer.decompose(W_numpy)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)

    phi_list_n = add_gaussian_noise_cpu(
        phi_list, noise_mean=0, noise_std=phase_noise_std, trunc_range=())

    if(protect_mask is not None):
        phi_list_n[protect_mask] = phi_list[protect_mask]

    phi_mat_n = vector_to_upper_triangle_cpu(phi_list_n)
    if(batch_mode):
        W_recon = decomposer.reconstruct_2_batch(delta_list, phi_mat_n)
    else:
        W_recon = decomposer.reconstruct_2(delta_list, phi_mat_n)

    W_recon = torch.from_numpy(W_recon).to(torch.float32).to(output_device)

    # res = check_equal_tensor(W, W_recon)
    # print("[I] checkEqual: ", res)
    # print("S:", Sigma)
    # print("delta_U:", delta_list_U)
    # print("delta_V:", delta_list_V)
    # print("W:", W)
    # print("W_rec:", W_recon)

    # bin=2**v_bit
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(W_numpy.reshape([-1]), bins=bin, range=[-1, 1])
    # plt.subplot(2,1,2)
    # plt.hist(W_recon.detach().cpu().numpy().reshape([-1]), bins=bin, range=[-1, 1])

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(phi_list_U.reshape([-1]), bins=bin, range=[-np.pi, np.pi])
    # plt.subplot(2,1,2)
    # plt.hist(phi_list_U_q.reshape([-1]), bins=bin, range=[-np.pi, np.pi])

    # v_max = np.sqrt(2*np.pi/gamma)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.hist(v_list_U.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.subplot(2,1,2)
    # plt.hist(v_list_U_q.reshape([-1]), bins=bin, range=[0, v_max])
    # plt.show()

    return W_recon


def circulant_multiply(c, x):
    """ Multiply circulant matrix with first column c by x
    Parameters:
        c: (n, )
        x: (batch_size, n) or (n, )
    Return:
        prod: (batch_size, n) or (n, )
    """
    return torch.irfft(complex_mult(torch.rfft(c, 1), torch.rfft(x, 1)), 1, signal_sizes=(c.shape[-1], ))


def calcDiagonalHessian(weight_dict, loss, model):
    model.zero_grad()
    hessian_dict = {}
    for name, weight in weight_dict.items():
        first_gradient = grad(loss, weight, create_graph=True)[0]
        second_gradient = grad(first_gradient.sum(),
                               weight, create_graph=True)[0]
        hessian_dict[name] = second_gradient.clone()
    model.zero_grad()
    return hessian_dict


def calcJacobian(weight_dict, loss, model):
    model.zero_grad()
    jacobian_dict = {}
    for name, weight in weight_dict.items():
        first_gradient = grad(loss, weight, create_graph=True)[0]
        jacobian_dict[name] = first_gradient.clone()
    model.zero_grad()
    return jacobian_dict


def calcSparsity(drop_masks, layers, device=torch.device("cuda")):
    n_drop = torch.zeros(1).to(device)
    n_params = torch.zeros(1).to(device)
    total_params = torch.zeros(1).to(device)

    for layer, lasso in drop_masks.items():
        mini_block = layers[layer].mini_block
        n_drop = torch.sum(1 - lasso)
        n_params += (lasso.size(0) * lasso.size(1) - n_drop) * mini_block
        total_params += lasso.size(0) * lasso.size(1) * mini_block
    sparsity = 1 - n_params.to(torch.float32) / total_params.to(torch.float32)
    return sparsity


def calcSoftNNHardware(layers):
    area_table = {"DC": 54.4*40.3,
                  "PS": 60.16*0.5,
                  "AT": 54.4*40.3,
                  "CB": 20*3.65,
                  "CR": 5.9*5.9}
    stat = {'n_coupler': 0,
            'n_phase_shifter': 0,
            'n_combiner': 0,
            'n_attenuator': 0,
            'n_crossing': 0,
            "n_params": 0,
            'area': 0,
            "sparsity": 0}

    fft_coupler_map = {2: 1, 4: 4, 8: 12}
    fft_phase_shifter_map = {2: 2, 4: 8, 8: 24}
    total_params = 0
    for layer, config in layers.items():
        in_channel = config["in_channel"]
        out_channel = config["out_channel"]
        mini_block = config["mini_block"]
        n_block = int(in_channel / mini_block) * \
            int(out_channel / mini_block) - config["n_block_remove"]

        stat['n_coupler'] += n_block * (fft_coupler_map[mini_block] * 2)
        stat['n_phase_shifter'] += n_block * \
            (fft_phase_shifter_map[mini_block] * 2 + mini_block)
        stat['n_attenuator'] += n_block * mini_block
        stat['n_combiner'] += (in_channel / mini_block - 1) * mini_block
        stat['n_crossing'] += (in_channel / mini_block - 1) * \
            mini_block * (mini_block - 1) / 2
        stat['n_params'] += n_block * mini_block
        total_params += int(in_channel * out_channel / mini_block)
    stat['area'] = stat['n_coupler']*area_table["DC"] + \
        stat['n_phase_shifter']*area_table["PS"] + \
        stat['n_attenuator']*area_table["AT"] + \
        stat['n_crossing']*area_table["CR"] + \
        stat['n_combiner']*area_table["CB"]
    # print((stat['n_crossing']*area_table["CR"] + stat['n_combiner']*area_table["CB"])/stat['area'])
    stat["sparsity"] = 1 - stat['n_params'] / total_params
    return stat


def calcMZIHardware(layers):
    stat = {'n_mzi': 0,
            'n_attenuator': 0,
            'n_params': 0,
            'area': 0}
    area_table = {"DC": 54.4*40.3,
                  "PS": 60.16*0.5,
                  "AT": 54.4*40.3,
                  "CB": 20*3.65,
                  "CR": 5.9*5.9}

    for layer, config in layers.items():
        in_channel = config["in_channel"]
        out_channel = config["out_channel"]
        stat['n_mzi'] += (in_channel * (in_channel - 1)) / \
            2 + (out_channel * (out_channel - 1)) / 2
        stat['n_attenuator'] += max(in_channel, out_channel)
        stat['n_params'] += in_channel * out_channel
    stat['area'] = (stat['n_mzi']*2+stat['n_attenuator']) * \
        area_table["DC"] + stat['n_mzi']*area_table["PS"]

    return stat


def calcSlimMZIHardware(layers):
    stat = {'n_mzi': 0,
            'n_attenuator': 0,
            'n_params': 0,
            'area': 0}
    area_table = {"DC": 54.4*40.3,
                  "PS": 60.16*0.5,
                  "AT": 54.4*40.3,
                  "CB": 20*3.65,
                  "CR": 5.9*5.9}

    for layer, config in layers.items():
        in_channel = config["in_channel"]
        out_channel = config["out_channel"]
        stat['n_mzi'] += (in_channel * (in_channel - 1)) / 2 + in_channel
        stat['n_attenuator'] += max(in_channel, out_channel)
        stat['n_params'] += in_channel * out_channel
    stat['area'] = (stat['n_mzi']*2+stat['n_attenuator']) * \
        area_table["DC"] + stat['n_mzi']*area_table["PS"]

    return stat


def calcMZILatency(layers):
    latency_table = {"DC": 54.4,  # directional coupler
                     "PS": 61.2,  # phase shifter
                     "AT": 54.4,  # attenuator
                     "CB": 20,  # combiner
                     "CR": 5.9}  # waveguide crossing
    total_length = 0
    n_DC = 0
    n_PS = 0
    for layer, config in layers.items():
        n = config["in_channel"]
        m = config["out_channel"]
        n_DC += 2 * n + 2 * m - 3
        n_PS += n + m - 2
    total_length = n_DC * latency_table["DC"] + n_PS * latency_table["PS"]
    total_latency = total_length / (1e6) / (3e8) * 1e12 * 2.3
    print(
        f"[I] Total latency for MZI Arch.: {total_length} um, {total_latency} ps")
    print(layers)
    return total_latency


def calcSlimMZILatency(layers):
    latency_table = {"DC": 54.4,  # directional coupler
                     "PS": 61.2,  # phase shifter
                     "AT": 54.4,  # attenuator
                     "CB": 20,  # combiner
                     "CR": 5.9}  # waveguide crossing
    total_length = 0
    n_DC = 0
    n_PS = 0
    for layer, config in layers.items():
        n = config["in_channel"]
        m = config["out_channel"]
        n_DC += 2 * n + 1
        n_PS += n
    total_length = n_DC * latency_table["DC"] + n_PS * latency_table["PS"]
    total_latency = total_length / (1e6) / (3e8) * 1e12 * 2.3
    print(
        f"[I] Total latency for Slim MZI Arch.: {total_length} um, {total_latency} ps")
    print(layers)
    return total_latency


def calcSoftNNLatency(layers):
    latency_table = {"DC": 54.4,  # directional coupler
                     "PS": 61.2,  # phase shifter
                     "AT": 54.4,  # attenuator
                     "CB": 20,  # combiner
                     "CR": 5.9}  # waveguide crossing
    total_length = 0
    n_DC = 0
    n_PS = 0
    n_CB = 0
    n_AT = 0
    n_CR = 0
    for layer, config in layers.items():
        n = config["in_channel"]
        m = config["out_channel"]
        k = config["mini_block"]
        n_DC += 2 * np.log2(k) + 1
        n_PS += 2 * np.log2(k) + 1
        n_CB = np.log2(n/k)
        n_AT = 1
        n_CR = 2*k-2*np.log2(k)-2+k*np.log2(n/k)
    total_length = n_DC * latency_table["DC"] + n_PS * latency_table["PS"] + n_CB * \
        latency_table["CB"] + n_AT * \
        latency_table["AT"] + n_CR * latency_table["CR"]
    total_latency = total_length / (1e6) / (3e8) * 1e12 * 2.3
    print(
        f"[I] Total latency for SoftNN Arch.: {total_length} um, {total_latency} ps")
    print(layers)
    return total_latency


class VowelRecog(data.Dataset):
    def __init__(self, path, mode='train'):
        self.path = path
        assert os.path.exists(path)
        assert mode in ['train', 'test']
        self.data, self.labels = self.load(mode=mode)

    def load(self, mode='train'):
        with open(f'{self.path}/{mode}.pt', 'rb') as f:
            data, labels = torch.load(f)
        return data, labels

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, item):
        return self.data[item], self.labels[item]


@profile
def test_tiling_usv():
    set_torch_deterministic()
    W = np.random.normal(loc=0, scale=0.2, size=[128, 128])

    # U,S,V = np.linalg.svd(W, full_matrices=True)
    # set_torch_stochastic()
    # U_n = add_gamma_noise_to_unitary_cpu(U, 4.36, 0.002).cpu().numpy()
    # set_torch_stochastic()
    # V_n = add_gamma_noise_to_unitary_cpu(V, 4.36, 0.002).cpu().numpy()
    # set_torch_deterministic()
    # W_n = np.dot(U_n, np.dot(np.diag(S), V_n))
    # return

    W_n = np.zeros_like(W)
    k = 8
    # pool = Pool(4)
    # tasks = [(i,j) for i in range(W.shape[0]//k) for j in range(W.shape[1]//k)]

    # def func(i,j):
    #     W_sub = W[i*k:(i+1)*k, j*k:(j+1)*k]
    #     U,S,V = np.linalg.svd(W_sub, full_matrices=True)
    #     set_torch_stochastic()
    #     # U_n = clip_voltage_of_unitary_cpu(U, 4.36, 0, 6).cpu().numpy()
    #     U_n = add_gamma_noise_to_unitary_cpu(U, 4.36, 0.002).cpu().numpy()
    #     set_torch_stochastic()
    #     # V_n = clip_voltage_of_unitary_cpu(V, 4.36, 0, 6).cpu().numpy()
    #     V_n = add_gamma_noise_to_unitary_cpu(V, 4.36, 0.002).cpu().numpy()
    #     set_torch_deterministic()
    #     W_sub_n = np.dot(U_n, np.dot(np.diag(S), V_n))
    #     W_n[i*k:(i+1)*k, j*k:(j+1)*k] = W_sub_n[:,:]
    # pool.map(lambda args: func(*args), tasks)
    # pool.close()

    for i in range(W.shape[0]//k):
        for j in range(W.shape[1]//k):
            W_sub = W[i*k:(i+1)*k, j*k:(j+1)*k]
            U, S, V = np.linalg.svd(W_sub, full_matrices=True)
            set_torch_stochastic()
            U_n = add_gamma_noise_to_unitary_cpu(U, 4.36, 0.002).cpu().numpy()
            set_torch_stochastic()
            V_n = add_gamma_noise_to_unitary_cpu(V, 4.36, 0.002).cpu().numpy()
            set_torch_deterministic()
            W_sub_n = np.dot(U_n, np.dot(np.diag(S), V_n))
            W_n[i*k:(i+1)*k, j*k:(j+1)*k] = W_sub_n[:, :]
    W_res = W - W_n
    print(
        f"tile W, {np.std(W_res), np.mean(W_res), np.max(W_res), np.min(W_res)}")


def cal_MZI_energy_efficiency(layers):
    latency = calcMZILatency(layers)
    print(latency, "ps")
    power = 0
    FLOPs = 0
    for layer, config in layers.items():
        n = config["in_channel"]
        m = config["out_channel"]
        power += (0.2 * 0.5 * 1e-8 * 1e6) * m
        FLOPs += 2*n*m
    eff = FLOPs / 1e9 / (power * latency * 1e-12)
    return eff

def cal_Slim_energy_efficiency(layers):
    latency = calcSlimMZILatency(layers)
    print(latency, "ps")
    power = 0
    FLOPs = 0
    for layer, config in layers.items():
        n = config["in_channel"]
        m = config["out_channel"]
        power += (0.2 * 0.5 * 1e-8 * 1e6) * m
        FLOPs += 2*n*m
    eff = FLOPs / 1e9 / (power * latency * 1e-12)
    return eff

def cal_FFT_energy_efficiency(layers):
    latency = calcSoftNNLatency(layers)
    print(latency, "ps")
    power = 0
    FLOPs = 0
    for layer, config in layers.items():
        n = config["in_channel"]
        m = config["out_channel"]
        power += (0.2 * 0.5 * 1e-8 * 1e6) * m
        FLOPs += 2*n*m
    eff = FLOPs / 1e9 / (power * latency * 1e-12)
    return eff



#########################
#          Plot         #
#########################

def plotGraph(X, Y):
    fig = plt.figure()
    return fig


def smooth_line(x, y, smoothness=0):
    assert 0 <= smoothness <= 1, f"[E] Only support smoothness within [0,1]"
    if(smoothness < 1e-3):
        return x, y

    N = len(x)
    N_new = int(N * (smoothness * 6 + 1))
    x_smooth = np.linspace(min(x), max(x), N_new)
    y_smooth = spline(x, y, x_smooth)
    return x_smooth.tolist(), y_smooth.tolist()


def draw_box_plot(data, ax, edge_color, fill_color, yrange):
    meanpointprops = dict(marker='D', markeredgecolor='black',
                          markerfacecolor='red', markersize=18)
    bp = ax.boxplot(data, showmeans=True, patch_artist=True,
                    meanprops=meanpointprops)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color, linewidth=13)
    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)
    # for flier in bp['fliers']:
    #	.set(marker='o', color='#e7298a', alpha=2)

    plt.setp(bp['fliers'], marker='+', markersize=18.0, color='k')
    plt.setp(bp['means'], markersize=15.0)

    # plt.setp(bp['whiskers'], color='k', linestyle='-')

    # plt.setp(bp['fliers'], )
    plt.yticks(np.arange(yrange[0], yrange[1], step=yrange[2]))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(80)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(80)


def draw_bar_plot(data, ax, barwidth, barcolor):
    bp = ax.bar(data["x"], data["y"], align='center',
                alpha=0.9, width=barwidth, color=barcolor)


def draw_line_plot(data, ax, linewidth, linecolor):
    bp = ax.plot(data["x"], data["y"], linewidth=linewidth, color=linecolor)


def batch_plot(type, raw_data, name, fig=None, ax=None, trace_color="#1871bf", xlabel="", ylabel="", xrange=[0, 1, 0.1], yrange=[0, 3, 0.5], figsize_pixels=[400, 300], dpi=600, fontsize=20, barwidth=0.1, linewidth=2, smoothness=0, turnoffy=False):
    assert type in {"box", "bar",
                    "line"}, f"[E] Only support box, bar and line chart"

    if(type == "box"):
        data = [i for i in raw_data.values()]
        draw_box_plot(data, ax, trace_color, 'white', yrange)

        xtl = [int(i) for i in raw_data.keys()]
        ax.set_xticklabels(xtl)
        # ax.set_yticklabels(ytl)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # if turnoffy is False:
        # ,   fontsize=12)
        plt.ylabel(ylabel,  fontsize=fontsize, fontweight='bold')
        plt.xlabel(xlabel,  fontsize=fontsize,
                   fontweight='bold')  # , fontsize=12)
        ax.grid(True, linewidth=11)
        # fig.set_size_inches(30, 30)
        fig.set_size_inches(
            figsize_pixels[0]/float(600), figsize_pixels[1]/float(600))

        [i.set_linewidth(11) for i in ax.spines.values()]
        plt.xticks(np.arange(xrange[0], xrange[1], step=xrange[2]))
        plt.yticks(np.arange(yrange[0], yrange[1], step=yrange[2]))

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
    elif(type == "bar"):
        # data := {"x" : [...], "y" : [...]}
        data = raw_data
        ax.grid(True, linewidth=linewidth)
        ax.set_axisbelow(True)
        draw_bar_plot(data, ax, barwidth, trace_color)

        # ax.set_yticklabels(ytl)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # if turnoffy is False:
        # ,   fontsize=12)
        plt.ylabel(ylabel,  fontsize=fontsize, fontweight='bold')
        plt.xlabel(xlabel,  fontsize=fontsize,
                   fontweight='bold')  # , fontsize=12)

        DPI = fig.get_dpi()
        fig.set_size_inches(
            figsize_pixels[0]/float(DPI), figsize_pixels[1]/float(DPI))

        [i.set_linewidth(linewidth) for i in ax.spines.values()]
        plt.xticks(np.arange(xrange[0], xrange[1], step=xrange[2]))
        plt.yticks(np.arange(yrange[0], yrange[1], step=yrange[2]))

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        plt.tight_layout()
    elif(type == "line"):
        data = raw_data
        if smoothness > 1e-2:
            x, y = smooth_line(
                raw_data["x"], raw_data["y"], smoothness=smoothness)
            data["x"] = x
            data["y"] = y
        ax.grid(True, linewidth=linewidth)
        ax.set_axisbelow(True)
        draw_line_plot(data, ax, linewidth, trace_color)

        # ax.set_yticklabels(ytl)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        # if turnoffy is False:
        # ,   fontsize=12)
        plt.ylabel(ylabel,  fontsize=fontsize, fontweight='bold')
        plt.xlabel(xlabel,  fontsize=fontsize,
                   fontweight='bold')  # , fontsize=12)

        DPI = fig.get_dpi()
        fig.set_size_inches(
            figsize_pixels[0]/float(DPI), figsize_pixels[1]/float(DPI))

        [i.set_linewidth(linewidth) for i in ax.spines.values()]
        plt.xticks(np.arange(xrange[0], xrange[1], step=xrange[2]))
        plt.yticks(np.arange(yrange[0], yrange[1], step=yrange[2]))

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(fontsize)
        plt.tight_layout()
    else:
        raise NotImplementedError

    # plt.savefig('boxp_' + name +'.pdf')

    # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    # plt.show()


def butterfly_layer_permutation(units: int, frequency: int):
    if units % 2:
        raise NotImplementedError('Odd input dimension case not yet implemented.')
    frequency = frequency
    unpermuted_indices = np.arange(units)
    num_splits = units // frequency
    total_num_indices = num_splits * frequency
    unpermuted_indices_remainder = unpermuted_indices[total_num_indices:]
    permuted_indices = np.hstack(
        [np.hstack([i, i + frequency] for i in range(frequency)) + 2 * frequency * split_num
         for split_num in range(num_splits // 2)] + [unpermuted_indices_remainder]
    )
    return permuted_indices.astype(np.int32)


def butterfly_permutation(num_layers: int):
    ordered_idx = np.arange(2 ** num_layers)
    permuted_idx = np.vstack(
        [butterfly_layer_permutation(2 ** num_layers, 2 ** layer) for layer in range(num_layers)]
    ).astype(np.int32)
    return np.vstack((ordered_idx.astype(np.int32),
                      permuted_idx[1:].astype(np.int32),
                      ordered_idx.astype(np.int32)))

if __name__ == "__main__":

    config1 = {"fc1": {"in_channel": 1024, "out_channel": 1024},
              "fc2": {"in_channel": 1024, "out_channel": 512},
              "fc3": {"in_channel": 512, "out_channel": 128}}
    config2 = {"fc1": {"in_channel": 1024, "out_channel": 1024, "mini_block": 16, "n_block_remove": 0},
              "fc2": {"in_channel": 1024, "out_channel": 512, "mini_block": 8, "n_block_remove": 0},
              "fc3": {"in_channel": 512, "out_channel": 128, "mini_block": 8, "n_block_remove": 0}}
    eff1 = cal_MZI_energy_efficiency(config1)
    eff2 = cal_Slim_energy_efficiency(config1)
    eff3 = cal_FFT_energy_efficiency(config2)
    print(eff1, eff2, eff3)
    exit(1)



    print(butterfly_permutation(3))
    exit(1)
    # W = gen_boolean_mask_cpu([8,8], 0.2)
    ud = RealUnitaryDecomposerBatch()



    # U, S, V = np.linalg.svd(W, full_matrices=True)
    # delta_list, phi_mat = ud.decompose(U)
    # print(phi_mat)

    N = 128
    delta_list = np.ones([N])
    phi_list = gen_boolean_mask_cpu([N*(N-1)//2], 0.3).astype(np.float64)*np.pi/3
    phi_mat = vector_to_upper_triangle_cpu(phi_list)
    U = ud.reconstruct_2(delta_list, phi_mat)
    phi_list = gen_boolean_mask_cpu([N*(N-1)//2], 0.6).astype(np.float64)*np.pi/3
    phi_mat = vector_to_upper_triangle_cpu(phi_list)
    V = ud.reconstruct_2(delta_list, phi_mat)
    S = np.diag(np.random.randn(N))
    W = U @ (S @ V)
    plt.figure()
    plt.hist(W.reshape([-1]), bins=128)
    plt.show()

    exit(1)


    ud = RealUnitaryDecomposerBatch()
    W = np.random.normal(loc=0, scale=0.2, size=[8, 8, 8, 8])
    U, S, V = np.linalg.svd(W, full_matrices=True)
    U_n = add_gamma_noise_to_unitary_cpu(U, 4.36, 0.002).detach().cpu().numpy()
    V_n = add_gamma_noise_to_unitary_cpu(V, 4.36, 0.002).detach().cpu().numpy()

    W_n = U_n @ (S[..., np.newaxis] * V_n)
    diff = np.mean((W_n-W)**2)
    print(diff)

    mask = np.random.choice(a=[False, True], size=(8, 8), p=[0.8, 0.2])
    W[mask, :, :] = 0
    U, S, V = np.linalg.svd(W, full_matrices=True)
    U_n = add_gamma_noise_to_unitary_cpu(U, 4.36, 0.002).detach().cpu().numpy()
    V_n = add_gamma_noise_to_unitary_cpu(V, 4.36, 0.002).detach().cpu().numpy()
    W_n = U_n @ (S[..., np.newaxis] * V_n)
    diff = np.mean((W_n-W)**2)
    print(diff)
    exit(1)

    W = np.dot(U, np.dot(S, V))

    print(ud.checkUnitary(W))
    exit()

    x = torch.randn(2, 3, 2, 2)
    count = 0
    print(x)
    new_x = merge_chunks(x)

    print(new_x)
    print(merge_chunks(x.numpy()))
    print(partition_chunks(new_x, 2))
    print(partition_chunks(new_x.numpy(), 2))
    exit(1)

    # from scipy.stats import truncnorm
    # # set_torch_deterministic()
    # w = truncnorm.rvs(-0.008263/0.002, 0.008263/0.002, loc=0, scale=0.002, size=[640,640], random_state=None)
    # print(w)
    # print(np.abs(w).max())
    # exit(1)
    set_torch_deterministic()
    W = np.random.normal(loc=0, scale=0.2, size=[32, 32])
    # W = projection_matrix_to_unitary_cpu(W)

    U, S, V = np.linalg.svd(W, full_matrices=True)
    # print(S)
    # S *=0.5
    # W = np.dot(U, np.dot(np.diag(S), V))
    print(np.std(W), np.mean(W))
    print("U", np.std(U), np.mean(U), np.max(U), np.min(U))
    print("V", np.std(V), np.mean(V), np.max(V), np.min(V))
    set_torch_stochastic()
    U_n = add_gamma_noise_to_unitary_cpu(U, 4.36, 0.002).cpu().numpy()
    set_torch_stochastic()
    V_n = add_gamma_noise_to_unitary_cpu(V, 4.36, 0.002).cpu().numpy()
    # U_n = quantize_voltage_of_unitary_cpu(U, 4, 4.36, 10.8).cpu().numpy()
    # V_n = quantize_voltage_of_unitary_cpu(V, 4, 4.36, 10.8).cpu().numpy()
    set_torch_deterministic()
    W_n = np.dot(U_n, np.dot(np.diag(S), V_n))
    U_res = U_n - U
    V_res = V_n - V
    W_res = W_n - W
    W_N = np.random.normal(loc=0, scale=np.std(W_res), size=[64, 64])
    U_N = np.random.normal(loc=0, scale=np.std(U_res), size=[64, 64])
    V_N = np.random.normal(loc=0, scale=np.std(V_res), size=[64, 64])
    W_N_f = torch.rfft(torch.from_numpy(W_N.reshape(
        [-1])), signal_ndim=1, onesided=False).numpy()
    W_res_f = torch.rfft(torch.from_numpy(W_res.reshape(
        [-1])), signal_ndim=1, onesided=False).numpy()

    print("W", np.std(W_res), np.mean(W_res), np.max(W_res), np.min(W_res))
    print("U", np.std(U_res), np.mean(U_res), np.max(U_res), np.min(U_res))
    print("V", np.std(V_res), np.mean(V_res), np.max(V_res), np.min(V_res))
    test_tiling_usv()
    exit(1)

    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.bar(np.arctan2(W_N_f[:, 1], W_N_f[:, 0]), np.sqrt(
    #     W_N_f[:, 0]**2 + W_N_f[:, 1]**2), width=0.1)
    # plt.subplot(2, 1, 2)
    # plt.bar(np.arctan2(W_res_f[:, 1], W_res_f[:, 0]), np.sqrt(
    #     W_res_f[:, 0]**2 + W_res_f[:, 1]**2), width=0.1)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(W_res.reshape([-1]), bins=256, range=[-0.5, 0.5])
    plt.subplot(2, 1, 2)
    plt.hist(W.reshape([-1]), bins=256, range=[-0.5, 0.5])

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(U_res.reshape([-1]), bins=256, range=[-0.25, 0.25])
    plt.subplot(2, 1, 2)
    plt.hist(U.reshape([-1]), bins=256, range=[-0.25, 0.25])

    # plt.figure()
    # plt.subplot(2, 1, 1)
    # plt.hist(V_res.reshape([-1]), bins=256, range=[-0.25, 0.25])
    # plt.subplot(2, 1, 2)
    # plt.hist(V_N.reshape([-1]), bins=256, range=[-0.25, 0.25])
    plt.show()
    exit(1)
    # export_traces_to_csv("./log/model_reg_q_6b.trace", "./log/model_reg_q-6b_lambda1-10_lambda2-1_clamp-1", fieldnames=None)
    # exit(1)
    torch.cuda.set_device(0)
    device = torch.device("cuda")
    # W = torch.Tensor(64, 64).to(device)
    # torch.nn.init.kaiming_normal_(W)
    # quantize_voltage_of_matrix_cpu(W, v_bit=4, v_pi=4.36)
    # exit(1)
    W_val = np.array([[0.4598, -0.2758, -0.5620,  0.6298],
                      [0.2988,  0.8944,  0.1421,  0.3005],
                      [-0.6394,  0.3277, -0.6954, -0.0103],
                      [-0.5390, -0.1282,  0.4244,  0.7161]]).astype(np.float64)
    W_q = np.array([[0.4598, -0.2758, -0.5620,  0.6298],
                    [0.2988,  0.8944,  0.1421,  0.3005],
                    [-0.6394,  0.3277, -0.6954, -0.0104],
                    [-0.5390, -0.1281,  0.4245,  0.7161]]).astype(np.float64)

    U, Sigma, V = np.linalg.svd(W_q, full_matrices=True)
    print(U)
    print(check_unitary_matrix(U))
    U += np.random.randn(4, 4)*0.1

    print(U)
    print(check_unitary_matrix(U))
    U_refine = projection_matrix_to_unitary_cpu(U)
    print(U_refine)
    print(check_unitary_matrix(U_refine))
    exit(1)

    Sigma, delta_U, phi_U, delta_V, phi_V = real_matrix_parametrization_cpu(
        W_q)
    # W_recon = real_matrix_reconstruction_cpu(Sigma, delta_U, phi_U, delta_V, phi_V)
    print(Sigma, delta_U, phi_U, delta_V, phi_V)
    Sigma, delta_U, phi_U, delta_V, phi_V = real_matrix_parametrization_cpu_ref(
        W_q)
    print(Sigma, delta_U, phi_U, delta_V, phi_V)
    exit(1)
    # U, Sigma, V = np.linalg.svd(W_val, full_matrices=True)
    # Sigma, delta_U, phi_U, delta_V, phi_V = real_matrix_parametrization_cpu(W_val)
    # W_recon = real_matrix_reconstruction_cpu(Sigma, delta_U, phi_U, delta_V, phi_V)
    print(Sigma, U, V)
    exit(1)

    # config = {"fc1": {"in_channel": 28**2, "out_channel": 80, "mini_block": 4, "n_block_remove": 1505},
    #                            "fc2": {"in_channel": 80, "out_channel": 10, "mini_block": 2, "n_block_remove": 9}}
    # config = {"fc1": {"in_channel": 8, "out_channel": 64, "mini_block": 8, "n_block_remove": 0},
    #           "fc2": {"in_channel": 64, "out_channel": 4, "mini_block": 2, "n_block_remove": 10}}
    # config = {"fc1": {"in_channel": 8, "out_channel": 24, "mini_block": 4, "n_block_remove": 0},
    #           "fc2": {"in_channel": 24, "out_channel": 4, "mini_block": 2, "n_block_remove": 3}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 60, "mini_block": 4, "n_block_remove": 307},
    #                            "fc2": {"in_channel": 60, "out_channel": 10, "mini_block": 2, "n_block_remove": 17}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 56, "mini_block": 4, "n_block_remove": 262},
    #           "fc2": {"in_channel": 56, "out_channel": 10, "mini_block": 2, "n_block_remove": 10}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 60, "mini_block": 4, "n_block_remove": 285},
    #           "fc2": {"in_channel": 60, "out_channel": 32, "mini_block": 4, "n_block_remove": 6},
    #           "fc3": {"in_channel": 32, "out_channel": 10, "mini_block": 2, "n_block_remove": 4}}
    # config = {"fc1": {"in_channel": 28**2, "out_channel": 64, "mini_block": 4, "n_block_remove": 1202},
    #           "fc2": {"in_channel": 64, "out_channel": 64, "mini_block": 4, "n_block_remove": 5},
    #           "fc3": {"in_channel": 64, "out_channel": 10, "mini_block": 2, "n_block_remove": 7}}
    # calcSoftNNLatency(config)
    # exit(1)

    # config = {"fc1": {"in_channel": 28**2, "out_channel": 20},
    #           "fc2": {"in_channel": 20, "out_channel": 10}}
    # # config = {"fc1": {"in_channel": 8, "out_channel": 18},
    # #           "fc2": {"in_channel": 18, "out_channel": 4}}
    # # config = {"fc1": {"in_channel": 8, "out_channel": 8},
    # #           "fc2": {"in_channel": 8, "out_channel": 4}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 16},
    #                   "fc2": {"in_channel": 16, "out_channel": 10}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 16},
    #           "fc2": {"in_channel": 16, "out_channel": 16},
    #           "fc3": {"in_channel": 16, "out_channel": 10}}
    # # config = {"fc1": {"in_channel": 28**2, "out_channel": 16},
    # #           "fc2": {"in_channel": 16, "out_channel": 34},
    # #           "fc3": {"in_channel": 34, "out_channel": 10}}
    # calcSlimMZILatency(config)
    # exit(1)

    # export_traces_to_csv("./log/model_SoftNN_80_10_np.trace", "./log/model_SoftNN_80_10_np.csv")
    # exit(1)
    # device = torch.device("cuda")
    # x = torch.Tensor(np.zeros([1, 4])).to(device)
    # w = torch.nn.Parameter(torch.Tensor([1, 2, 3, 4]).to(device))
    # torch.nn.init.uniform_(x, 0.0, 1.0)
    # c = Krylov(shift, w)
    # print(c)
    # print(torch.nn.functional.linear(x, c, None))
    # print(circulant_multiply(w, x))
    # sc = ThresholdScheduler(0, 50, 0.002, 0.02)
    # config = {"fc1": {"in_channel": 28**2, "out_channel": 80, "mini_block": 4, "n_block_remove": 1505},
    #                            "fc2": {"in_channel": 80, "out_channel": 10, "mini_block": 2, "n_block_remove": 9}}
    # config = {"fc1": {"in_channel": 28**2, "out_channel": 256, "mini_block": 4, "n_block_remove": 6701},
    #           "fc2": {"in_channel": 256, "out_channel": 10, "mini_block": 2, "n_block_remove": 58}}
    # config = {"fc1": {"in_channel": 28**2, "out_channel": 512, "mini_block": 4, "n_block_remove": 11112},
    #           "fc2": {"in_channel": 512, "out_channel": 10, "mini_block": 2, "n_block_remove": 72}}
    # config = {"fc1": {"in_channel": 28**2, "out_channel": 1024, "mini_block": 4, "n_block_remove": 24813},
    #           "fc2": {"in_channel": 1024, "out_channel": 10, "mini_block": 2, "n_block_remove": 300}}
    config = {"fc1": {"in_channel": 28**2, "out_channel": 1024, "mini_block": 8, "n_block_remove": 5198},
              "fc2": {"in_channel": 1024, "out_channel": 10, "mini_block": 2, "n_block_remove": 326}}
    # config = {"fc1": {"in_channel": 8, "out_channel": 64, "mini_block": 8, "n_block_remove": 0},
    #           "fc2": {"in_channel": 64, "out_channel": 4, "mini_block": 2, "n_block_remove": 10}}
    # config = {"fc1": {"in_channel": 8, "out_channel": 24, "mini_block": 4, "n_block_remove": 0},
    #           "fc2": {"in_channel": 24, "out_channel": 4, "mini_block": 2, "n_block_remove": 3}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 60, "mini_block": 4, "n_block_remove": 307},
    #           "fc2": {"in_channel": 60, "out_channel": 10, "mini_block": 2, "n_block_remove": 17}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 56, "mini_block": 4, "n_block_remove": 262},
    #           "fc2": {"in_channel": 56, "out_channel": 10, "mini_block": 2, "n_block_remove": 10}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 60, "mini_block": 4, "n_block_remove": 285},
    #           "fc2": {"in_channel": 60, "out_channel": 32, "mini_block": 4, "n_block_remove": 6},
    #           "fc3": {"in_channel": 32, "out_channel": 10, "mini_block": 2, "n_block_remove": 4}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 256, "mini_block": 4, "n_block_remove": 1190},
    #           "fc2": {"in_channel": 256, "out_channel": 10, "mini_block": 2, "n_block_remove": 89}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 512, "mini_block": 4, "n_block_remove": 2153},
    #           "fc2": {"in_channel": 512, "out_channel": 10, "mini_block": 2, "n_block_remove": 191}}
    # config = {"fc1": {"in_channel": 28**2, "out_channel": 64, "mini_block": 4, "n_block_remove": 1202},
    #           "fc2": {"in_channel": 64, "out_channel": 64, "mini_block": 4, "n_block_remove": 5},
    #           "fc3": {"in_channel": 64, "out_channel": 10, "mini_block": 2, "n_block_remove": 7}}
    stat = calcSoftNNHardware(config)
    calcSoftNNLatency(config)
    print("SoftNN w/ prune:", config)
    print(stat)
    print()
    config["fc1"]["n_block_remove"] = 0
    config["fc2"]["n_block_remove"] = 0
    stat = calcSoftNNHardware(config)
    print("SoftNN w/o prune:", config)
    print(stat)
    # exit(1)

    # config = {"fc1": {"in_channel": 28**2, "out_channel": 20},
    #           "fc2": {"in_channel": 20, "out_channel": 10}}
    # config = {"fc1": {"in_channel": 28**2, "out_channel": 48},
    #           "fc2": {"in_channel": 48, "out_channel": 10}}
    # config = {"fc1": {"in_channel": 28**2, "out_channel": 64},
    #           "fc2": {"in_channel": 64, "out_channel": 10}}
    # config = {"fc1": {"in_channel": 28**2, "out_channel": 72},
    #           "fc2": {"in_channel": 72, "out_channel": 10}}
    # config = {"fc1": {"in_channel": 28**2, "out_channel": 256},
    #           "fc2": {"in_channel": 256, "out_channel": 10}}
    config = {"fc1": {"in_channel": 28**2, "out_channel": 400},
              "fc2": {"in_channel": 400, "out_channel": 10}}
    # config = {"fc1": {"in_channel": 8, "out_channel": 18},
    #           "fc2": {"in_channel": 18, "out_channel": 4}}
    # config = {"fc1": {"in_channel": 8, "out_channel": 8},
    #           "fc2": {"in_channel": 8, "out_channel": 4}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 16},
    #                            "fc2": {"in_channel": 16, "out_channel": 10}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 16},
    #           "fc2": {"in_channel": 16, "out_channel": 16},
    #           "fc3": {"in_channel": 16, "out_channel": 10}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 64},
    #           "fc2": {"in_channel": 64, "out_channel": 10}}
    # config = {"fc1": {"in_channel": 14**2, "out_channel": 128},
    #           "fc2": {"in_channel": 128, "out_channel": 10}}
    # config = {"fc1": {"in_channel": 28**2, "out_channel": 16},
    #           "fc2": {"in_channel": 16, "out_channel": 34},
    #           "fc3": {"in_channel": 34, "out_channel": 10}}
    print()
    stat = calcMZIHardware(config)
    calcMZILatency(config)
    print("mzi:", config)
    print(stat)

    print()
    stat = calcSlimMZIHardware(config)
    calcSlimMZILatency(config)
    print("slim mzi:", config)
    print(stat)
