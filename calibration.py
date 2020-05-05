import torch
import numpy as np
from scipy import stats as scistats
def DP_correction():
    b = 8
    # coeff2 = sum(i*j*c for i in range(0, 2**b) for j in range(0, 2**b)) / 2**(2*b)
    # print(coeff2)
    N = 16*9

    x = (torch.randint(0, int(2**b), size=[N]).to(float).cuda() / (2**b-1))
    w = (torch.randint(0, int(2**b), size=[N]).to(float).cuda() / (2**b-1))
    phi = torch.zeros_like(x).normal_(mean=3.1415926535/2, std=0.1).cuda()
    disks = 1 - torch.zeros(N,2).normal_(mean=0, std=0.03).cuda().abs()
    print(x, w, disks)
    sin_phi = phi.sin()
    beta0 = disks[...,0].sum()
    beta1 = disks[...,1].sum()
    correction_add = -1/4*((x**2).mean()**2+(w**2).mean())*(beta0-beta1)
    print(correction_add.data.item())
    beta2 = (sin_phi * (disks[...,0]+disks[...,1])/2).mean()
    res1 = torch.dot(w, x)
    res2 = torch.dot(w*sin_phi*(disks[...,0]+disks[...,1])/2, x) + ((w**2+x**2)*(disks[...,0]-disks[...,1])).sum()/4
    res3 = (res2 + correction_add) / beta2
    res4 = res2 / beta2
    print("real=", res1.data.item())
    print("noisy=", res2.data.item())
    print("full_cali=",res3.data.item())
    print("mul_cali=",res4.data.item())


def DP_correction_2(phase_noise_std=0.1, disk_noise_std=0.05):
    b = 2
    # coeff2 = sum(i*j*c for i in range(0, 2**b) for j in range(0, 2**b)) / 2**(2*b)
    # print(coeff2)
    N = 4

    x = (torch.randint(0, int(2**b), size=[N]).to(float).cuda() / (2**b-1))
    w = (torch.randint(0, int(2**b), size=[N]).to(float).cuda() / (2**b-1))
    # x = torch.Tensor([1,0,1,1]).float().cuda()
    # w = torch.Tensor([1,0,1,1]).float().cuda()
    phi = torch.zeros_like(x).normal_(mean=-np.pi/2, std=phase_noise_std).cuda().clamp(-np.pi/2-3*phase_noise_std, -np.pi/2+3*phase_noise_std)
    print(x)
    print(w)
    phi[phi.size(0)//2:] *= -1
    disks = 1 - torch.zeros(N,2).normal_(mean=0, std=disk_noise_std).cuda().clamp(-3*disk_noise_std, 3*disk_noise_std).abs()
    print(phi)
    print(disks)
    sin_phi = phi.sin()
    # beta0 = scistats.gmean(disks[...,0].cpu().numpy())
    # beta1 = scistats.gmean(disks[...,1].cpu().numpy())

    beta2 = (sin_phi.abs() * (disks[...,0]+disks[...,1])/2).mean() / (disks.mean())
    # beta3 = scistats.gmean(sin_phi.abs().cpu().numpy())
    wb = w.clone()
    wb[w.size(0)//2:] *= -1
    #sort


    res1 = torch.dot(wb, x)
    rail_0 = (disks[...,0]*(w**2 - 2*w*x*sin_phi + x**2)).sum() / 4
    rail_1 = (disks[...,1]*(w**2 + 2*w*x*sin_phi + x**2)).sum() / 4
    print(2*rail_0, 2*rail_1, 2*(rail_0-rail_1))
    res2 = rail_0 - rail_1
    res22 = torch.dot(-w*sin_phi*(disks[...,0]+disks[...,1])/2, x) + ((w**2+x**2)*(disks[...,0]-disks[...,1])).sum()/4
    res3 = (rail_0 / disks[...,0].mean() - rail_1 / disks[...,1].mean())/ beta2

    for start, end in [(0, w.size(0)//2), (w.size(0)//2, w.size(0))]:
        W = w[start:end]
        print(W)
        _, w_indices = W.abs().sort()
        print(w_indices)
        # _, d_indices = (disks[start:end, 0] - disks[start:end,1]).abs().sort(descending=True)
        _, d_indices = (phi[start:end].sin().abs()*(disks[start:end, 0] + disks[start:end,1])).sort(descending=False)
        print(d_indices)
        print(phi[start:end])
        phi[start:end][w_indices] = phi[start:end][d_indices]
        print(phi[start:end])
        disks[start:end, :][w_indices,:] = disks[start:end,:][d_indices,:] #

    sin_phi = phi.sin()
    rail_0 = (disks[...,0]*(w**2 - 2*w*x*sin_phi + x**2)).sum() / 4
    rail_1 = (disks[...,1]*(w**2 + 2*w*x*sin_phi + x**2)).sum() / 4
    res4 = rail_0 - rail_1
    res5 = (rail_0 / disks[...,0].mean() - rail_1 / disks[...,1].mean())/ beta2

    # res4 = (rail_0 / beta0 - rail_1 / beta1)/ beta3

    print("real=", res1.data.item())
    print("noisy=", res2.data.item())
    print("noisy=", res22.data.item())
    print("full_cali=",res3.data.item())
    print("map=",res4.data.item())
    print("map+cali=",res5.data.item())

    return (res2 - res1).abs().data.item(), (res3 - res1).abs().data.item()


def DP_correction_simulation(phase_noise_std=0.1, disk_noise_std=0.05):
    b = 2
    # coeff2 = sum(i*j*c for i in range(0, 2**b) for j in range(0, 2**b)) / 2**(2*b)
    # print(coeff2)
    N = 4

    x = torch.tensor([0.0667,0.2,0.7333,0.4667]).cuda()
    w = torch.tensor([0.5333,0.1333,0.7333,0.2667]).cuda()

    phi = torch.tensor([-np.pi/2,-np.pi/2,np.pi/2,np.pi/2]).cuda()
    print(x)
    print(w)

    disks = torch.zeros(N,2).cuda()
    disks.data[:,:] = 0.933254300796991
    print(phi)
    print(disks)
    sin_phi = phi.sin()
    # beta0 = scistats.gmean(disks[...,0].cpu().numpy())
    # beta1 = scistats.gmean(disks[...,1].cpu().numpy())

    beta2 = (sin_phi.abs() * (disks[...,0]+disks[...,1])/2).mean() / (disks.mean())
    # beta3 = scistats.gmean(sin_phi.abs().cpu().numpy())
    wb = w.clone()
    wb[w.size(0)//2:] *= -1
    #sort


    res1 = torch.dot(wb, x)
    rail_0 = (disks[...,0]*(w**2 - 2*w*x*sin_phi + x**2)).sum() / 4
    rail_1 = (disks[...,1]*(w**2 + 2*w*x*sin_phi + x**2)).sum() / 4
    print(2*rail_0, 2*rail_1, 2*(rail_0-rail_1))
    res2 = rail_0 - rail_1
    res22 = torch.dot(-w*sin_phi*(disks[...,0]+disks[...,1])/2, x) + ((w**2+x**2)*(disks[...,0]-disks[...,1])).sum()/4
    res3 = (rail_0 / disks[...,0].mean() - rail_1 / disks[...,1].mean())/ beta2

    sin_phi = phi.sin()
    rail_0 = (disks[...,0]*(w**2 - 2*w*x*sin_phi + x**2)).sum() / 4
    rail_1 = (disks[...,1]*(w**2 + 2*w*x*sin_phi + x**2)).sum() / 4
    res4 = rail_0 - rail_1
    res5 = (rail_0 / disks[...,0].mean() - rail_1 / disks[...,1].mean())/ beta2

    # res4 = (rail_0 / beta0 - rail_1 / beta1)/ beta3

    print("real=", res1.data.item()*2)
    print("noisy=", res2.data.item()*2)
    print("noisy=", res22.data.item()*2)
    print("full_cali=",res3.data.item()*2)

    return (res2 - res1).abs().data.item(), (res3 - res1).abs().data.item()

def test_DP_calibration():
    base_mean = []
    base_std = []
    mean = []
    std = []
    for phase_std, disk_std in [(0.02, 0.01), (0.04, 0.02), (0.06, 0.03), (0.08, 0.04), (0.1, 0.05), (0.12, 0.06)]:
        base_res = []
        res = []
        print(phase_std, disk_std)
        for i in range(500):
            base_error, error = DP_correction_2(phase_std, disk_std)
            base_res.append(base_error)
            res.append(error)
        mean.append(np.mean(res))
        std.append(np.std(res))
        base_mean.append(np.mean(base_res))
        base_std.append(np.std(base_res))
    for m, s, bm, bs in zip(mean, std, base_mean, base_std):
        print(m, s, bm, bs)

def EC_correction():
    b = 8
    c = 1/((2**b-1)**2)
    ii = torch.arange(0, 2**b)
    jj = torch.arange(0, 2**b)
    grid_x, grid_y = torch.meshgrid(ii, jj)
    coeff = (grid_x.double() * grid_y.double() * c).sum() / 2**(2*b)
    print(coeff)
    # coeff2 = sum(i*j*c for i in range(0, 2**b) for j in range(0, 2**b)) / 2**(2*b)
    # print(coeff2)
    N = 16*9

    x = torch.randint(0, int(2**b), size=[N]).to(float).cuda() / (2**b-1)
    w = torch.randint(0, int(2**b), size=[N]).to(float).cuda() / (2**b-1)
    phi = torch.zeros_like(x).normal_(mean=-3.1415926535/2, std=0.1).cuda()
    disks = 1 - torch.zeros(N,2).normal_(mean=0, std=0.05).cuda().abs()
    print(x, w)
    sin_phi = phi.sin()
    beta = (disks[...,1]*sin_phi).mean() / disks[...,1].mean()
    res1 = -(w**2 - 2*w*x + x**2).sum()
    res2 = -(disks[...,1]*(w**2 + 2*w*x*sin_phi + x**2)).sum()
    # res3 = res2 + 2 * N * coeff * (1+beta)
    res3 = res2/disks[...,1].mean() + 2 * N * coeff * (1+beta)
    print(res1)
    print(res2)
    print(res3)

if __name__ == "__main__":
    # DP_correction_2(0.0001,0.00001)
    DP_correction_simulation(0.0001,0.00001)
    # DP_correction()
    # EC_correction()
    # test_DP_calibration()
