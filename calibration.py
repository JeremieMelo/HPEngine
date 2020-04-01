import torch

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


def DP_correction_2():
    b = 8
    # coeff2 = sum(i*j*c for i in range(0, 2**b) for j in range(0, 2**b)) / 2**(2*b)
    # print(coeff2)
    N = 16*9

    x = (torch.randint(0, int(2**b), size=[N]).to(float).cuda() / (2**b-1))
    w = (torch.randint(0, int(2**b), size=[N]).to(float).cuda() / (2**b-1))
    phi = torch.zeros_like(x).normal_(mean=-3.1415926535/2, std=0.1).cuda()
    disks = 1 - torch.zeros(N,2).normal_(mean=0, std=0.05).cuda().abs()
    print(x, w, disks)
    sin_phi = phi.sin()
    beta0 = disks[...,0].sum()
    beta1 = disks[...,1].sum()

    beta2 = -(sin_phi * (disks[...,0]+disks[...,1])/2).mean() / (disks.mean())
    res1 = torch.dot(w, x)
    rail_0 = (disks[...,0]*(w**2 - 2*w*x*sin_phi + x**2)).sum() / 4
    rail_1 = (disks[...,1]*(w**2 + 2*w*x*sin_phi + x**2)).sum() / 4
    res2 = rail_0 - rail_1
    res3 = (rail_0 / disks[...,0].mean() - rail_1 / disks[...,1].mean())/ beta2

    print("real=", res1.data.item())
    print("noisy=", res2.data.item())
    print("full_cali=",res3.data.item())


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
    DP_correction_2()
    # EC_correction()
