#!/usr/bin/env python
# coding=UTF-8
'''
@Author: Jake Gu
@Date: 2019-03-18 21:21:06
@LastEditTime: 2019-07-28 00:32:28
'''
import time
from functools import partial
from multiprocessing.dummy import Pool

import numpy as np
import tensorflow as tf
import torch
from numba import jit
from scipy.stats import ortho_group, unitary_group
from torch import nn

import utils

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.enable_eager_execution(config=config)


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


class RealUnitaryDecomposer(object):
    timer = False

    def __init__(self, min_err=1e-7, timer=False, determine=False):
        self.min_err = min_err
        self.timer = timer
        self.determine = determine

    def buildPlaneUnitary(self, p, q, phi, N, transpose=True):
        assert N > 0 and isinstance(
            N, int), "[E] Matrix size must be positive integer"
        assert isinstance(p, int) and isinstance(q,
                                                 int) and 0 <= p < q < N, "[E] Integer value p and q must satisfy p < q"
        assert (isinstance(phi, float) or isinstance(phi, int)
                ), "[E] Value phi must be of type float or int"

        U = np.eye(N)
        c = np.cos(phi)
        s = np.sin(phi)

        U[p, p] = U[q, q] = c
        U[q, p] = s if not transpose else -s
        U[p, q] = -s if not transpose else s

        return U

    def calPhi_determine(self, u1, u2, is_first_col=False):
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -np.pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * np.pi if u2 > min_err else 0.5 * np.pi
        else:
            # solve the equation: u'_1n=0
            if(is_first_col):
                phi = np.arctan2(-u2, u1)  # 4 quadrant
            else:
                phi = np.arctan(-u2/u1)

    def calPhi_nondetermine(self, u1, u2, is_first_col=False):
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -np.pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * np.pi if u2 > min_err else 0.5 * np.pi
        else:
            # solve the equation: u'_1n=0
            phi = np.arctan2(-u2, u1)  # 4 quadrant

        print(phi, u1, u2)

        return phi

    # @profile(timer=timer)
    def decomposeKernel(self, U, dim, phi_list=None):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[0]
        if(phi_list is None):
            phi_list = np.zeros(dim, dtype=np.float64)

        calPhi = self.calPhi_determine if self.determine else self.calPhi_nondetermine
        for i in range(N - 1):
            # with utils.TimerCtx() as t:
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            phi = calPhi(u1, u2, is_first_col=(i == 0))

            # print("calPhi:", t.interval)
            # with utils.TimerCtx() as t:
            phi_list[i] = phi
            p, q = 0, N - i - 1
            # U = np.dot(U, self.buildPlaneUnitary(p=p, q=q, phi=phi, N=N, transpose=True))
            c, s = np.cos(phi), np.sin(phi)
            row_p, row_q = U[:, p], U[:, q]
            U[:, p], U[:, q] = row_p * c - row_q * s, row_p * s + row_q * c
            print(U)
            # U[:, p], U[:, q] = U[:, p] * c - U[:, q] * s, U[:, p] * s + U[:, q] * c
            # print("rotate:", t.interval)

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    @profile(timer=timer)
    def decompose(self, U_ori):
        U = U_ori.copy()
        N = U.shape[0]
        assert N > 0 and U.shape[0] == U.shape[1], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros([N, N], dtype=np.float64)
        sigma_mat = np.zeros([N, N], dtype=np.float64)
        delta_list = np.zeros(N, dtype=np.float64)

        for i in range(N - 1):
            # U, phi_list = self.decomposeKernel(U, dim=N)
            # phi_mat[i, :] = phi_list
            U, _ = self.decomposeKernel(U, dim=N, phi_list=phi_mat[i, :])
            delta_list[i] = U[0, 0]
            U = U[1:, 1:]
        else:
            delta_list[-1] = U[-1, -1]

        # print(f'[I] Decomposition done')
        return delta_list, phi_mat

    @profile(timer=timer)
    def reconstruct(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        for i in range(N):
            for j in range(N - i - 1):
                phi = phi_mat[i, j]
                Ur = np.dot(self.buildPlaneUnitary(
                    i, N - j - 1, phi, N, transpose=False), Ur)

        D = np.diag(delta_list)
        Ur = np.dot(D, Ur)
        print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    @profile(timer=timer)
    def reconstruct_2(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        # cos_phi = np.cos(phi_list)
        # sin_phi = np.sin(phi_list)
        # print(phi_list)
        # print(cos_phi)
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        for i in range(N-1):
            for j in range(N - i - 1):
                # with utils.TimerCtx() as t:
                # phi = phi_mat[i, j]
                # c = np.cos(phi)
                # s = np.sin(phi)
                # index = int((2 * N - i - 1) * i / 2 + j)
                # phi = phi_list[index]
                c, s = phi_mat_cos[i, j], phi_mat_sin[i, j]

                # print("cos:", t.interval)
                # c = cos_phi[count]
                # s = sin_phi[count]
                # count += 1
                # with utils.TimerCtx() as t:
                p = i
                q = N - j - 1
                # Ur_new = Ur_old.clone()
                Ur[p, :], Ur[q, :] = Ur[p, :] * c - \
                    Ur[q, :] * s, Ur[p, :] * s + Ur[q, :] * c
                # print("rotate:", t.interval)
        D = np.diag(delta_list)
        Ur = np.dot(D, Ur)
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    def checkIdentity(self, M):
        return (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))

    def checkUnitary(self, U):
        M = np.dot(U, U.T)
        # print(M)
        return self.checkIdentity(M)

    def checkEqual(self, M1, M2):
        return (M1.shape[0] == M2.shape[0]) and (M1.shape[1] == M2.shape[1]) and np.allclose(M1, M2)

    def genRandomOrtho(self, N):
        U = ortho_group.rvs(N)
        print(
            f'[I] Generate random {N}*{N} unitary matrix, check unitary: {self.checkUnitary(U)}')
        return U

    def toDegree(self, M):
        return np.degrees(M)


class RealUnitaryDecomposerPyTorch(object):
    timer = True

    def __init__(self, device="cuda", min_err=1e-6, timer=False, use_multithread=False, n_thread=8):
        self.min_err = min_err
        self.timer = timer
        self.device = torch.device(device)
        self.use_multithread = use_multithread
        self.n_thread = n_thread
        self.pi = torch.Tensor([np.pi]).to(self.device)
        self.zero = torch.Tensor([0]).to(self.device)
        if(self.use_multithread):
            self.pool = Pool(self.n_thread)
        else:
            self.pool = None

    def buildPlaneUnitary(self, p, q, phi, N, transpose=True):

        U = torch.eye(N, device=self.device)
        c = torch.cos(phi)
        s = torch.sin(phi)

        U[p, p] = U[q, q] = c
        U[q, p] = s if not transpose else -s
        U[p, q] = -s if not transpose else s

        return U

    def calPhi(self, u1, u2):
        u1_abs, u2_abs = torch.abs(u1), torch.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = self.zero
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = self.zero if u1 > min_err else -self.pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = torch.Tensor([-0.5]).to(self.device) * \
                self.pi if u2 > min_err else 0.5 * self.pi
        else:
            # solve the equation: u'_1n=0
            phi = torch.atan2(-u2, u1)  # 4 quadrant

        if len(phi.size()) == 0:
            phi = phi.unsqueeze(0)
        return phi

    # @profile(timer=timer)
    def decomposeKernel(self, U, dim):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.size(0)
        phi_list = torch.zeros((dim)).to(self.device)

        for i in range(N - 1):
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            phi = self.calPhi(u1, u2)
            assert len(phi.size()) > 0, f"{phi}"
            phi_list[i] = phi
            p, q = 0, N - i - 1
            U = torch.mm(U, self.buildPlaneUnitary(
                p=p, q=q, phi=phi, N=N, transpose=True))

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    # @profile(timer=timer)
    def decompose(self, U):
        N = U.size(0)

        phi_mat = torch.zeros((N, N)).to(self.device)
        delta_list = torch.zeros((N)).to(self.device)

        for i in range(N - 1):
            U, phi_list = self.decomposeKernel(U, dim=N)
            phi_mat[i, :] = phi_list
            delta_list[i] = U[0, 0]
            U = U[1:, 1:]
        else:
            delta_list[-1] = U[-1, -1]

        # print(f'[I] Decomposition done')
        return delta_list, phi_mat

    @profile(timer=timer)
    def reconstruct(self, delta_list, phi_mat):
        N = delta_list.size(0)
        Ur = nn.init.eye_(torch.empty((N, N))).to(self.device)

        # reconstruct from right to left as in the book chapter
        for i in range(N):
            for j in range(N - i - 1):
                phi = phi_mat[i, j]
                Ur = torch.mm(self.buildPlaneUnitary(
                    i, N - j - 1, phi, N, transpose=False), Ur)

        D = torch.diag(delta_list).to(self.device)
        Ur = torch.mm(D, Ur)
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    @profile(timer=timer)
    def reconstruct_2(self, delta_list, phi_list):
        N = delta_list.size(0)
        Ur = nn.init.eye_(torch.empty((N, N))).to(self.device)
        Ur_new = nn.init.eye_(torch.empty((N, N))).to(self.device)

        # reconstruct from right to left as in the book chapter
        if(self.use_multithread == False):
            for i in range(N):
                for j in range(N - i - 1):
                    # phi = phi_mat[i, j]
                    index = int((2 * N - i - 1) * i / 2 + j)
                    phi = phi_list[index]
                    c = torch.cos(phi)
                    s = torch.sin(phi)
                    p = i
                    q = N - j - 1
                    # Ur_new = Ur_old.clone()
                    Ur_new[p, ...], Ur_new[q, ...] = Ur[p, ...] * c - \
                        Ur[q, ...] * s, Ur[p, ...] * s + Ur[q, ...] * c
                    Ur = Ur_new.clone()
                    # Ur = torch.mm(self.buildPlaneUnitary(i, N - j - 1, phi, N, transpose=False), Ur)

        else:

            PlaneUnitary_list = [(i, j, phi_list[int((2 * N - i - 1) * i / 2 + j)])
                                 for i in range(N) for j in range(N - i - 1)]

            PlaneUnitary_list = self.pool.map(lambda args: self.buildPlaneUnitary(
                args[0], N - args[1] - 1, args[2], N, transpose=False), PlaneUnitary_list)

            PlaneUnitary_list = torch.stack(PlaneUnitary_list, dim=0)
            n_planes = PlaneUnitary_list.size(0)
            log2_n_planes = int(np.log2(n_planes))
            n_iters = log2_n_planes if(
                2**log2_n_planes == n_planes) else log2_n_planes + 1

            for _ in range(n_iters):
                even_batch = PlaneUnitary_list[::2]
                odd_batch = PlaneUnitary_list[1::2]
                if(odd_batch.size(0) < even_batch.size(0)):
                    odd_batch = torch.cat([odd_batch, torch.eye(
                        N).to(self.device).unsqueeze(0)], dim=0)
                PlaneUnitary_list = torch.bmm(odd_batch, even_batch)

            Ur = PlaneUnitary_list.squeeze(0)

        D = torch.diag(delta_list).to(self.device)
        Ur = torch.mm(D, Ur)
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    def prunePhases(self, phi_mat, epsilon=1e-4):
        N = phi_mat.size(0)
        is_close_to_0 = phi_mat.abs() < epsilon
        is_close_to_90 = torch.abs(phi_mat - self.pi/2) < epsilon
        is_close_to_180 = torch.abs(phi_mat - self.pi) < epsilon
        is_close_to_270 = torch.abs(phi_mat + self.pi/2) < epsilon
        print(is_close_to_0.sum()-N*(N+1)/2)
        print(is_close_to_90.sum())
        print(is_close_to_180.sum())
        print(is_close_to_270.sum())
        phi_mat[is_close_to_0] = self.zero
        phi_mat[is_close_to_90] = self.pi/2
        phi_mat[is_close_to_180] = self.pi
        phi_mat[is_close_to_270] = -self.pi/2
        n_prune = is_close_to_0.sum() + is_close_to_90.sum() + is_close_to_180.sum() + \
            is_close_to_270.sum() - N*(N+1)/2
        return n_prune

    def checkIdentity(self, M):
        I = nn.init.eye_(torch.empty((N, N))).to(self.device)
        return (M.size(0) == M.size(1)) and torch.allclose(M, I, rtol=1e-04, atol=1e-07)

    def checkUnitary(self, U):
        M = torch.mm(U, U.t())
        print(M)
        return self.checkIdentity(M)

    def checkEqual(self, M1, M2):
        return (M1.size() == M2.size()) and torch.allclose(M1, M2, rtol=1e-04, atol=1e-07)

    def genRandomOrtho(self, N):
        U = torch.Tensor(ortho_group.rvs(N)).to(self.device)
        print(
            f'[I] Generate random {N}*{N} unitary matrix, check unitary: {self.checkUnitary(U)}')
        return U

    def toDegree(self, M):
        return M * 180 / self.pi


class RealUnitaryDecomposerBatch(object):
    timer = False

    def __init__(self, min_err=1e-7, timer=False, determine=False):
        self.min_err = min_err
        self.timer = timer
        self.determine = determine

    def buildPlaneUnitary(self, p, q, phi, N, transpose=True):
        assert N > 0 and isinstance(
            N, int), "[E] Matrix size must be positive integer"
        assert isinstance(p, int) and isinstance(q,
                                                 int) and 0 <= p < q < N, "[E] Integer value p and q must satisfy p < q"
        assert (isinstance(phi, float) or isinstance(phi, int)
                ), "[E] Value phi must be of type float or int"

        U = np.eye(N)
        c = np.cos(phi)
        s = np.sin(phi)

        U[p, p] = U[q, q] = c
        U[q, p] = s if not transpose else -s
        U[p, q] = -s if not transpose else s

        return U

    def calPhi_batch_determine(self, u1: np.ndarray, u2: np.ndarray, is_first_col=False) -> np.ndarray:
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err
        cond1 = u1_abs < min_err
        cond2 = u2_abs < min_err
        cond1_n = ~cond1
        cond2_n = ~cond2
        if(is_first_col):
            phi = np.where(cond1 & cond2, 0,
                           np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                    np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan2(-u2, u1))))
        else:
            phi = np.where(cond1 & cond2, 0,
                           np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                    np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan(-u2/u1))))
        return phi

    def calPhi_batch_nondetermine(self, u1: np.ndarray, u2: np.ndarray, is_first_col=False) -> np.ndarray:
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err
        cond1 = u1_abs < min_err
        cond2 = u2_abs < min_err
        cond1_n = ~cond1
        cond2_n = ~cond2
        phi = np.where(cond1 & cond2, 0,
                       np.where(cond1_n & cond2, np.where(u1 > min_err, 0, -pi),
                                np.where(cond1 & cond2_n, np.where(u2 > min_err, -0.5*pi, 0.5*pi), np.arctan2(-u2, u1))))

        return phi

    def calPhi_determine(self, u1, u2, is_first_col=False):
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * pi if u2 > min_err else 0.5 * pi
        else:
            # solve the equation: u'_1n=0
            if(is_first_col):
                phi = np.arctan2(-u2, u1)  # 4 quadrant4
            else:
                phi = np.arctan(-u2/u1)
            # phi = np.arctan(-u2/u1)
        # print(phi, u1, u2)

        # cond = ((u1_abs < min_err) << 1) | (u2_abs < min_err)
        # if(cond == 0):
        #     phi = np.arctan2(-u2, u1)
        # elif(cond == 1):
        #     phi = 0 if u1 > min_err else -np.pi
        # elif(cond == 2):
        #     phi = -0.5 * np.pi if u2 > min_err else 0.5 * np.pi
        # else:
        #     phi = 0
        # phi = [np.arctan2(-u2, u1), 0 if u1 > min_err else -np.pi, -0.5 * np.pi if u2 > min_err else 0.5 * np.pi, 0][cond]

        return phi

    def calPhi_nondetermine(self, u1, u2, is_first_col=False):
        pi = np.pi
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1 > min_err else -pi
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * pi if u2 > min_err else 0.5 * pi
        else:
            # solve the equation: u'_1n=0
            phi = np.arctan2(-u2, u1)  # 4 quadrant4

        return phi

 # @profile(timer=timer)
    def decomposeKernel_batch(self, U: np.ndarray, dim, phi_list=None) -> (np.ndarray, np.ndarray):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[-1]
        if(phi_list is None):
            phi_list = np.zeros(list(U.shape[:-2])+[dim], dtype=np.float64)

        calPhi_batch = self.calPhi_batch_determine if self.determine else self.calPhi_batch_nondetermine
        for i in range(N - 1):
            # with utils.TimerCtx() as t:
            u1, u2 = U[..., 0, 0], U[..., 0, N - 1 - i]
            phi = calPhi_batch(u1, u2, is_first_col=(i == 0))

            # print("calPhi:", t.interval)
            # with utils.TimerCtx() as t:
            phi_list[..., i] = phi

            p, q = 0, N - i - 1
            # U = np.dot(U, self.buildPlaneUnitary(p=p, q=q, phi=phi, N=N, transpose=True))
            c, s = np.cos(phi)[..., np.newaxis], np.sin(phi)[..., np.newaxis]
            col_p, col_q = U[..., :, p], U[..., :, q]
            U[..., :, p], U[..., :, q] = col_p * \
                c - col_q * s, col_p * s + col_q * c
            # U[:, p], U[:, q] = U[:, p] * c - U[:, q] * s, U[:, p] * s + U[:, q] * c
            # print("rotate:", t.interval)

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    # @profile(timer=timer)

    def decomposeKernel(self, U, dim, phi_list=None):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[0]
        if(phi_list is None):
            phi_list = np.zeros(dim, dtype=np.float64)

        calPhi = self.calPhi_determine if self.determine else self.calPhi_nondetermine
        for i in range(N - 1):
            # with utils.TimerCtx() as t:
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            phi = calPhi(u1, u2, is_first_col=(i == 0))

            # print("calPhi:", t.interval)
            # with utils.TimerCtx() as t:
            phi_list[i] = phi
            p, q = 0, N - i - 1
            # U = np.dot(U, self.buildPlaneUnitary(p=p, q=q, phi=phi, N=N, transpose=True))
            c, s = np.cos(phi), np.sin(phi)
            row_p, row_q = U[:, p], U[:, q]
            U[:, p], U[:, q] = row_p * c - row_q * s, row_p * s + row_q * c
            # U[:, p], U[:, q] = U[:, p] * c - U[:, q] * s, U[:, p] * s + U[:, q] * c
            # print("rotate:", t.interval)

        # print(f'[I] Decomposition kernel done')
        return U, phi_list

    @profile(timer=timer)
    def decompose_batch(self, U: np.ndarray) -> (np.ndarray, np.ndarray):
        N = U.shape[-1]
        assert N > 0 and U.shape[-1] == U.shape[-2], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros(U.shape, dtype=np.float64)
        delta_list = np.zeros(U.shape[:-1], dtype=np.float64)

        for i in range(N - 1):
            # U, phi_list = self.decomposeKernel(U, dim=N)
            # phi_mat[i, :] = phi_list
            U, _ = self.decomposeKernel_batch(
                U, dim=N, phi_list=phi_mat[..., i, :])
            delta_list[..., i] = U[..., 0, 0]
            U = U[..., 1:, 1:]
        else:
            delta_list[..., -1] = U[..., -1, -1]

        # print(f'[I] Decomposition done')
        return delta_list, phi_mat

    @profile(timer=timer)
    def decompose(self, U):
        N = U.shape[0]
        assert N > 0 and U.shape[0] == U.shape[1], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros([N, N], dtype=np.float64)
        delta_list = np.zeros(N, dtype=np.float64)

        for i in range(N - 1):
            # U, phi_list = self.decomposeKernel(U, dim=N)
            # phi_mat[i, :] = phi_list
            U, _ = self.decomposeKernel(U, dim=N, phi_list=phi_mat[i, :])
            delta_list[i] = U[0, 0]
            U = U[1:, 1:]
        else:
            delta_list[-1] = U[-1, -1]

        # print(f'[I] Decomposition done')
        return delta_list, phi_mat

    @profile(timer=timer)
    def reconstruct(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        for i in range(N):
            for j in range(N - i - 1):
                phi = phi_mat[i, j]
                Ur = np.dot(self.buildPlaneUnitary(
                    i, N - j - 1, phi, N, transpose=False), Ur)

        D = np.diag(delta_list)
        Ur = np.dot(D, Ur)
        print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    @profile(timer=timer)
    def reconstruct_2_batch(self, delta_list: np.ndarray, phi_mat: np.ndarray) -> np.ndarray:
        N = delta_list.shape[-1]
        Ur = tf.eye(
            num_rows=N,
            num_columns=N,
            batch_shape=delta_list.shape[:-1],
            dtype=tf.dtypes.float64
        ).numpy()

        # reconstruct from right to left as in the book chapter
        # cos_phi = np.cos(phi_list)
        # sin_phi = np.sin(phi_list)
        # print(phi_list)
        # print(cos_phi)
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        for i in range(N):
            for j in range(N - i - 1):
                # with utils.TimerCtx() as t:
                # phi = phi_mat[i, j]
                # c = np.cos(phi)
                # s = np.sin(phi)
                # index = int((2 * N - i - 1) * i / 2 + j)
                # phi = phi_list[index]
                c, s = phi_mat_cos[..., i, j:j+1], phi_mat_sin[..., i, j:j+1]

                # print("cos:", t.interval)
                # c = cos_phi[count]
                # s = sin_phi[count]
                # count += 1
                # with utils.TimerCtx() as t:
                p = i
                q = N - j - 1
                # Ur_new = Ur_old.clone()
                Ur[..., p, :], Ur[..., q, :] = Ur[..., p, :] * c - \
                    Ur[..., q, :] * s, Ur[..., p, :] * s + Ur[..., q, :] * c
                # print("rotate:", t.interval)

        D = tf.linalg.diag(
            diagonal=delta_list
        )

        # Ur = np.dot(D, Ur)
        Ur = tf.matmul(D, Ur).numpy()
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    @profile(timer=timer)
    def reconstruct_2(self, delta_list, phi_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        # cos_phi = np.cos(phi_list)
        # sin_phi = np.sin(phi_list)
        # print(phi_list)
        # print(cos_phi)
        # count = 0
        phi_mat_cos = np.cos(phi_mat)
        phi_mat_sin = np.sin(phi_mat)
        for i in range(N):
            for j in range(N - i - 1):
                # with utils.TimerCtx() as t:
                # phi = phi_mat[i, j]
                # c = np.cos(phi)
                # s = np.sin(phi)
                # index = int((2 * N - i - 1) * i / 2 + j)
                # phi = phi_list[index]
                c, s = phi_mat_cos[i, j], phi_mat_sin[i, j]

                # print("cos:", t.interval)
                # c = cos_phi[count]
                # s = sin_phi[count]
                # count += 1
                # with utils.TimerCtx() as t:
                p = i
                q = N - j - 1
                # Ur_new = Ur_old.clone()
                Ur[p, :], Ur[q, :] = Ur[p, :] * c - \
                    Ur[q, :] * s, Ur[p, :] * s + Ur[q, :] * c
                # print("rotate:", t.interval)
        D = np.diag(delta_list)
        Ur = np.dot(D, Ur)
        # print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    def checkIdentity(self, M):
        return (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))

    def checkUnitary(self, U):
        M = np.dot(U, U.T)
        # print(M)
        return self.checkIdentity(M)

    def checkEqual(self, M1, M2):
        return (M1.shape == M2.shape) and np.allclose(M1, M2)

    def genRandomOrtho(self, N):
        U = ortho_group.rvs(N)
        print(
            f'[I] Generate random {N}*{N} unitary matrix, check unitary: {self.checkUnitary(U)}')
        return U

    def toDegree(self, M):
        return np.degrees(M)


class ComplexUnitaryDecomposer(object):
    timer = True

    def __init__(self, min_err=1e-6, timer=False):
        self.min_err = min_err
        self.timer = timer

    def buildPlaneUnitary(self, p, q, phi, sigma, N, transpose=True):
        assert N > 0 and isinstance(
            N, int), "[E] Matrix size must be positive integer"
        assert isinstance(p, int) and isinstance(q,
                                                 int) and 0 <= p < q < N, "[E] Integer value p and q must satisfy p < q"
        assert (isinstance(phi, float) or isinstance(phi, int)) and (
            isinstance(sigma, float) or isinstance(sigma,
                                                   int)), "[E] Value phi and sigma must be of type float or int"

        U = np.eye(N, dtype=complex)
        c = np.cos(phi)
        s = np.sin(phi)

        U[p, p] = U[q, q] = c
        U[q, p] = s * np.exp(1j * sigma) if not transpose else - \
            s * np.exp(1j * sigma)
        U[p, q] = -s * \
            np.exp(-1j * sigma) if not transpose else s * np.exp(-1j * sigma)

        return U

    def calPhiSigma(self, u1, u2):
        u1_abs, u2_abs = np.abs(u1), np.abs(u2)
        u1_real, u2_img = np.real(u1), np.imag(u2)
        min_err = self.min_err

        if u1_abs < min_err and u2_abs < min_err:
            phi = 0
            sigma = 0
        elif u1_abs >= min_err and u2_abs < min_err:
            phi = 0 if u1_real > -min_err else -np.pi
            sigma = 0
        elif u1_abs < min_err and u2_abs >= min_err:
            phi = -0.5 * np.pi if u2_img < min_err else 0.5 * np.pi
            sigma = 0
        else:
            # solve the equation: u'_1n=0
            # first cal sigma, make its range -pi/2 <= sigma <= pi/2
            sigma = np.angle(u2.conj() / u1.conj())
            if (sigma > np.pi / 2):  # solution sigma need to be adjusted by +pi or -pi, then case 2 for phi
                sigma -= np.pi
                if (u1_real > -min_err):
                    phi = np.arctan(u2_abs / u1_abs)  # 1 quadrant
                else:
                    phi = np.arctan(u2_abs / u1_abs) - np.pi  # 3 quadrant
            # solution sigma need to be adjusted by +pi or -pi, then case 2 for phi
            elif (sigma < -np.pi / 2):
                sigma += np.pi
                if (u1_real > -min_err):
                    phi = np.arctan(u2_abs / u1_abs)  # 1 quadrant
                else:
                    phi = np.arctan(u2_abs / u1_abs) - np.pi  # 3 quadrant
            else:  # solution sigma satisfies its range, then case 1 for phi
                if (u1_real > -min_err):
                    phi = np.arctan(-u2_abs / u1_abs)  # 4 quadrant
                else:
                    phi = np.arctan(-u2_abs / u1_abs) + np.pi  # 2 quadrant

        return phi, sigma

    @profile(timer=timer)
    def decomposeKernel(self, U, dim):
        '''return U^(N-1); (phi_1,...,phi_N-2); (sigma_1,...,sigma_N-2)'''
        N = U.shape[0]
        phi_list = np.zeros(dim, dtype=np.float64)
        sigma_list = np.zeros(dim, dtype=np.float64)

        for i in range(N - 1):
            u1, u2 = U[0, 0], U[0, N - 1 - i]
            phi, sigma = self.calPhiSigma(u1, u2)
            phi_list[i], sigma_list[i] = phi, sigma
            p, q = 0, N - i - 1
            U = np.dot(U, self.buildPlaneUnitary(
                p=p, q=q, phi=phi, sigma=sigma, N=N, transpose=True))

        print(f'[I] Decomposition kernel done')
        return U, phi_list, sigma_list

    @profile(timer=timer)
    def decompose(self, U):
        N = U.shape[0]
        assert N > 0 and U.shape[0] == U.shape[1], '[E] Input matrix must be square and N > 0'

        phi_mat = np.zeros([N, N], dtype=np.float64)
        sigma_mat = np.zeros([N, N], dtype=np.float64)
        delta_list = np.zeros(N, dtype=complex)

        for i in range(N - 1):
            U, phi_list, sigma_list = self.decomposeKernel(U, dim=N)
            phi_mat[i, :], sigma_mat[i, :] = phi_list, sigma_list
            delta_list[i] = U[0, 0]
            U = U[1:, 1:]
        else:
            delta_list[-1] = U[-1, -1]

        print(f'[I] Decomposition done')
        return delta_list, phi_mat, sigma_mat

    @profile(timer=timer)
    def reconstruct(self, delta_list, phi_mat, sigma_mat):
        N = delta_list.shape[0]
        Ur = np.identity(N)

        # reconstruct from right to left as in the book chapter
        for i in range(N):
            for j in range(N - i - 1):
                phi, sigma = phi_mat[i, j], sigma_mat[i, j]
                Ur = np.dot(self.buildPlaneUnitary(
                    i, N - j - 1, phi, sigma, N, transpose=False), Ur)

        D = np.diag(delta_list)
        Ur = np.dot(D, Ur)
        print(f'[I] Reconstruct {N}*{N} unitary matrix done')

        return Ur

    def checkIdentity(self, M):
        return (M.shape[0] == M.shape[1]) and np.allclose(M, np.eye(M.shape[0]))

    def checkUnitary(self, U):
        M = np.dot(U, U.conj().T)
        return self.checkIdentity(M)

    def checkEqual(self, M1, M2):
        return (M1.shape[0] == M2.shape[0]) and (M1.shape[1] == M2.shape[1]) and np.allclose(M1, M2)

    def genRandomUnitary(self, N):
        U = unitary_group.rvs(N)
        print(
            f'[I] Generate random {N}*{N} unitary matrix, check unitary: {self.checkUnitary(U)}')
        return U

    def toDegree(self, M):
        return np.degrees(M)


def sparsifyDecomposition(phi_mat, sigma_mat=None, row_preserve=1):
    phi_mat[row_preserve:, ...] = 0
    if(sigma_mat):
        sigma_mat[row_preserve:, ...] = 0
    print(f'[I] Sparsify decomposition')


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


if __name__ == '__main__':

    udb = RealUnitaryDecomposerBatch()

    U = udb.genRandomOrtho(4)
    delta_list, phi_mat = udb.decompose(U)
    phi_mat[phi_mat > 1] = 0
    print(phi_mat)
    U_recon = udb.reconstruct_2(delta_list, phi_mat).astype(np.float32)
    delta_list, phi_mat = udb.decompose(U_recon)
    print(phi_mat)
    exit(1)

    W = np.random.randn(32, 32, 8, 8)
    W2 = W.copy()
    with utils.TimerCtx() as t:
        delta_list, phi_mat = udb.decompose_batch(W)
    print(t.interval)

    with utils.TimerCtx() as t:
        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                delta, phi = udb.decompose(W2[i, j, ...])
                res = udb.checkEqual(delta_list[i, j, ...], delta)
                # if(not res):
                #     print(i,j, "delta")

                # res = udb.checkEqual(phi_mat[i,j,...], phi)
                # if(not res):
                #     # pass
                #     print(phi_mat[i,j,...], phi)
    print(t.interval)
    Ur = udb.reconstruct_2_batch(delta_list, phi_mat)
    with utils.TimerCtx() as t:
        Ur = udb.reconstruct_2_batch(delta_list, phi_mat)
    print(t.interval)

    with utils.TimerCtx() as t:
        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                Ur2 = udb.reconstruct_2(
                    delta_list[i, j, ...], phi_mat[i, j, ...])
                # res = udb.checkEqual(Ur[i,j,...], Ur2)
                # if(not res):
                #     print("error")
    print(t.interval)

    exit(1)

    # u1 = np.random.randn(64*64)
    # u2 = np.random.randn(64*64)
    # phi = ud.calPhi_batch(u1,u2)
    # # print(phi)
    # phi_list = []
    # with utils.TimerCtx() as t:
    #     for i in range(64*64):
    #         phi_list.append(ud.calPhi(u1[i], u2[i]))
    # print(t.interval*1000, "ms")
    # phi_list = np.array(phi_list)
    # print(ud.checkEqual(phi, phi_list))

    # print(phi_list)
    exit(1)
    # np.set_printoptions(threshold=np.inf)
    # N = 5
    # ud = ComplexUniaryDecomposer(timer=True)
    # U = ud.genRandomUnitary(N)
    # # print(f'[I] Original:\n{U}')

    # UP, phi_list, sigma_list = ud.decomposeKernel(U, dim=N)
    # print(f'[I] U Prime:\n{UP}')
    # print(f'[I] check U Prime is unitary:{ud.checkUnitary(UP)}')
    # print(f'[I] phi_list:\n{phi_list}')
    # print(f'[I] sigma_list:\n{sigma_list}')

    # delta_list, phi_mat, sigma_mat = ud.decompose(U)
    # print(f'[I] delta_list:\n{delta_list}')
    # print(f'[I] phi_mat:\n{ud.toDegree(phi_mat)}')
    # print(f'[I] sigma_mat:\n{sigma_mat}')

    # sparsifyDecomposition(phi_mat, sigma_mat, row_preserve=1)
    # Ur = ud.reconstruct(delta_list, phi_mat, sigma_mat)
    # #print(f'[I] Reconstructed:\n{Ur}')
    # with fullprint(threshold=None, linewidth=150, precision=2):
    #     print(Ur)
    # print(f'[I] Check reconstruction is unitary: {ud.checkUnitary(Ur)}')
    # # print(f'[I] Check reconstruction is equal to original: {ud.checkEqual(Ur, U)}')

    # real matrix parameterization
    N = 64
    ud = RealUnitaryDecomposer(timer=True)
    U = ud.genRandomOrtho(N)
    U_cuda = torch.from_numpy(U).cuda().float()
    # U = np.eye(4)
    print(U)

    delta_list, phi_mat = ud.decompose(U)

    print(f'[I] delta_list:\n{delta_list}')
    print(f'[I] phi_mat:\n{phi_mat}')

    ud_2 = RealUnitaryDecomposerPyTorch(
        timer=True, use_multithread=False, n_thread=24)
    # U = ud.genRandomOrtho(N)
    # print(f'[I] Original:\n{U}')

    # UP, phi_list, sigma_list = ud.decomposeKernel(U, dim=N)
    # print(f'[I] U Prime:\n{UP}')
    # print(f'[I] check U Prime is unitary:{ud.checkUnitary(UP)}')
    # print(f'[I] phi_list:\n{phi_list}')
    # print(f'[I] sigma_list:\n{sigma_list}')

    delta_list_2, phi_mat_2 = ud_2.decompose(U_cuda)
    print(f'[I] delta_list CUDA:\n{delta_list_2}')
    print(f'[I] phi_mat CUDA:\n{phi_mat_2}')

    # phi_list_2 = torch.zeros(N*(N-1)//2).cuda()
    # phi_list = np.zeros([N*(N-1)//2])
    # for i in range(N):
    #     for j in range(N-i-1):
    #         phi_list_2[int((2 * N - i - 1) * i / 2 + j)] = phi_mat_2[i, j]
    #         phi_list[int((2 * N - i - 1) * i / 2 + j)] = phi_mat[i, j]
    # print("phi_list:", phi_list_2)
    # print("phi_list:", phi_list)

    # sparsifyDecomposition(phi_mat, row_preserve=1)
    Ur = ud.reconstruct(delta_list, phi_mat)
    print(f'[I] Reconstructed:\n{Ur}')
    print(f'[I] Check reconstruction is unitary: {ud.checkUnitary(Ur)}')
    Ur = ud.reconstruct_2(delta_list, phi_mat)
    print(f'[I] Reconstructed:\n{Ur}')
    print(f'[I] Check reconstruction is unitary: {ud.checkUnitary(Ur)}')

    Ur_2 = ud_2.reconstruct(delta_list_2, phi_mat_2)
    print(f'[I] Check reconstruction is unitary: {ud_2.checkUnitary(Ur_2)}')
    Ur_2 = ud_2.reconstruct_2(delta_list_2, phi_list_2)
    print(f'[I] Check reconstruction is unitary: {ud_2.checkUnitary(Ur_2)}')

    # with fullprint(threshold=None, linewidth=150, precision=4):
    #     print(Ur)
    print(Ur_2)
    print(f'[I] Check reconstruction is unitary: {ud.checkUnitary(Ur)}')
    # print(f'[I] Check reconstruction is equal to original: {ud.checkEqual(Ur, U)}')
