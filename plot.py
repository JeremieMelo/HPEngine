from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
try:
    from matrix_parametrization import RealUnitaryDecomposer
except:
    print("No unitary parametrization module found")
from utils import (batch_plot, draw_line_plot, phase_to_voltage_cpu, set_torch_deterministic,
                   upper_triangle_to_vector_cpu, vector_to_upper_triangle_cpu,
                   voltage_quantize_fn_cpu, voltage_to_phase_cpu)


def test_voltage_quantization(dirname="./figs"):
    set_torch_deterministic()
    ud = RealUnitaryDecomposer()
    N = 64
    U = ud.genRandomOrtho(N)
    bit = 6
    gamma = np.pi / 4.36**2
    delta_list, phi_mat = ud.decompose(U)
    phi_list = upper_triangle_to_vector_cpu(phi_mat)
    v_list = phase_to_voltage_cpu(phi_list, gamma)
    hist = np.histogram(np.reshape(U, [-1]), bins=16)
    print("unitary")
    # [print(i/N/N) for i in hist[0]]
    # print()
    # [print((hist[1][i]+hist[1][i+1])/2) for i in range(len(hist[1])-1)]
    fig, ax = plt.subplots()
    batch_plot("line", {"x": [(hist[1][i]+hist[1][i+1])/2 for i in range(len(hist[1])-1)], "y": [i/N/N for i in hist[0]]}, "test", fig, ax,
               trace_color="#2678B2", xlabel="Value", ylabel="Percentage", xrange=[-0.50, 0.5001, 0.5], yrange=[0, 0.201, 0.1], barwidth=0.1/6, linewidth=2, fontsize=20, figsize_pixels=[400, 300], smoothness=0.8)
    plt.savefig(f"{dirname}/UnitaryHist.pdf")

    print("phase")
    phi_list[phi_list > 0] -= 2*np.pi
    phi_list = -phi_list
    hist = np.histogram(phi_list, bins=16)
    # [print(i/(N*(N-1)//2)) for i in hist[0]]
    # print()
    # [print((hist[1][i]+hist[1][i+1])/2) for i in range(len(hist[1])-1)]
    fig, ax = plt.subplots()
    batch_plot("line", {"x": [(hist[1][i]+hist[1][i+1])/2 for i in range(len(hist[1])-1)], "y": [i/(N*(N-1)//2) for i in hist[0]]}, "test", fig, ax,
               trace_color="#2678B2", xlabel="Phase Lag (rad)", ylabel="Percentage", xrange=[0, 6.02, 2], yrange=[0, 0.4, 0.1], barwidth=0.1, linewidth=2, fontsize=20, figsize_pixels=[400, 300], smoothness=0.8)
    plt.savefig(f"{dirname}/PhaseHist.pdf")

    print("voltage")
    hist = np.histogram(v_list, bins=16)
    # [print(i/(N*(N-1)//2)) for i in hist[0]]
    # print()
    # [print((hist[1][i]+hist[1][i+1])/2) for i in range(len(hist[1])-1)]
    fig, ax = plt.subplots()
    batch_plot("line", {"x": [(hist[1][i]+hist[1][i+1])/2 for i in range(len(hist[1])-1)], "y": [i/(N*(N-1)//2) for i in hist[0]]}, "test", fig, ax,
               trace_color="#2678B2", xlabel="Voltage (V)", ylabel="Percentage", xrange=[0, 6.02, 2], yrange=[0, 0.5, 0.2], barwidth=0.1, linewidth=2, fontsize=20, figsize_pixels=[400, 300], smoothness=0.6)
    plt.savefig(f"{dirname}/VoltageHist.pdf")

    quantizer = voltage_quantize_fn_cpu(bit, 4.36, 10.8)
    v_list_q = quantizer(v_list)

    print("voltage Q")
    hist = Counter(v_list_q)
    # [print(i/(N*(N-1)//2)) for i in hist.values()]
    # print()
    # [print(i) for i in hist]
    fig, ax = plt.subplots()
    batch_plot("bar", {"x": [i for i in hist], "y": [i/(N*(N-1)//2) for i in hist.values()]}, "test", fig, ax,
               trace_color="#BF5700", xlabel="Voltage (V)", ylabel="Percentage", xrange=[0, 6.02, 2], yrange=[0, 0.3, 0.1], barwidth=0.1, linewidth=2, fontsize=20, figsize_pixels=[400, 300])
    plt.savefig(f"{dirname}/Quant{bit}VoltageBar.pdf")

    phi_list_q = voltage_to_phase_cpu(v_list_q, gamma)
    print("Phase Q")
    phi_list_q[phi_list_q > 0] -= 2*np.pi
    phi_list_q = -phi_list_q
    hist = Counter(phi_list_q)
    # [print(i/(N*(N-1)//2)) for i in hist.values()]
    # print()
    # [print(i) for i in hist]
    fig, ax = plt.subplots()
    batch_plot("bar", {"x": [i for i in hist], "y": [i/(N*(N-1)//2) for i in hist.values()]}, "test", fig, ax,
               trace_color="#BF5700", xlabel="Phase Lag (rad)", ylabel="Percentage", xrange=[0, 6.02, 2], yrange=[0, 0.4, 0.1], barwidth=0.1, linewidth=2, fontsize=20, figsize_pixels=[400, 300])
    plt.savefig(f"{dirname}/Quant{bit}PhaseBar.pdf")

    phi_mat_q = vector_to_upper_triangle_cpu(phi_list_q)
    U_q = ud.reconstruct_2(delta_list, phi_mat_q)

    print("unitary Q")
    hist = np.histogram(np.reshape(U_q, [-1]), bins=16)
    # [print(i/(N*(N-1)//2)) for i in hist[0]]
    # print()
    # [print((hist[1][i]+hist[1][i+1])/2) for i in range(len(hist[1])-1)]
    fig, ax = plt.subplots()
    batch_plot("line", {"x": [(hist[1][i]+hist[1][i+1])/2 for i in range(len(hist[1])-1)], "y": [i/N/N for i in hist[0]]}, "test", fig, ax,
               trace_color="#BF5700", xlabel="Value", ylabel="Percentage", xrange=[-0.50, 0.5001, 0.5], yrange=[0, 0.201, 0.1], barwidth=0.1/6, linewidth=2, fontsize=20, figsize_pixels=[400, 300], smoothness=0.8)
    plt.savefig(f"{dirname}/Quant{bit}UnitaryHist.pdf")


def plot_phase_voltage_curve(dirname="./figs"):
    fig, ax = plt.subplots()
    gamma = np.pi / 4.36**2
    v_2pi = np.sqrt(2*np.pi/gamma)
    v = np.linspace(0, v_2pi, num=100)
    phi = gamma * v**2
    data = {"x": v, "y": phi}
    batch_plot("line", data, "test", fig, ax, trace_color="#698236", xlabel="Voltage (V)", ylabel="Phase Lag (rad)", xrange=[
               0, v_2pi, 2], yrange=[0, 8, 3.14], barwidth=0.1/6, linewidth=2, fontsize=20, figsize_pixels=[400, 400], smoothness=0.3)

    phi_plus = (gamma * 1.15) * v**2
    data = {"x": v, "y": phi_plus}
    batch_plot("line", data, "test", fig, ax, trace_color="#698236", xlabel="Voltage (V)", ylabel="Phase Lag (rad)", xrange=[
               0, v_2pi, 2], yrange=[0, 8, 3.14], barwidth=0.1/6, linewidth=1, fontsize=20, figsize_pixels=[400, 400], smoothness=0.3)
    ax.fill_between(v, phi, phi_plus, color="#D0DFB3")

    phi_minus = (gamma * 0.85) * v**2
    data = {"x": v, "y": phi_minus}
    batch_plot("line", data, "test", fig, ax, trace_color="#698236", xlabel="Voltage (V)", ylabel="Phase Lag (rad)", xrange=[
               0, v_2pi, 2], yrange=[0, 8, 3.14], barwidth=0.1/6, linewidth=1, fontsize=20, figsize_pixels=[800, 800], smoothness=0.3)
    ax.fill_between(v, phi, phi_minus, color="#D0DFB3")
    ax.margins(0.0001, x=None, y=None, tight=True)
    plt.savefig(f"{dirname}/PhaseVoltageCurve.pdf")

def plot_md_curve(dirname="./figs"):
    fig, ax = plt.subplots()
    with open(f"{dirname}/add_drop_phi_0.txt", 'r') as f:
        lines = f.readlines()
        length = 23188-5000
        lines1 = np.array(list(map(lambda x: x[:-1].split(", "), lines[5000:5000+length]))).astype(np.float32)
        lines2 = np.array(list(map(lambda x: x[:-1].split(", "), lines[5500:5500+length]))).astype(np.float32)
        wl, t1, t2 = lines1[:,0], lines1[:, 1]**2, lines2[:, 1]**2
        t = np.vstack([t1,t2]).transpose([1,0])
        data = {"x": wl, "y": t1}
        batch_plot("line", data, "test", fig, ax,
               trace_color="#2678B2", xlabel="Wavelength (nm)", ylabel="Transmission", xrange=[np.min(wl), np.max(wl), 2.3], yrange=[0, 1.001, 0.2], barwidth=0.1/6, linewidth=2, fontsize=14, figsize_pixels=[400, 300], smoothness=0)
        # ax.plot(wl, t1, 2, "#2678B2")
        # lines1 = np.array(list(map(lambda x: x[:-1].split(", "), lines[6000:6000+length]))).astype(np.float32)
        # wl, t = lines1[:,0], lines1[:, 1]**2
        data = {"x": wl, "y": t2}
        draw_line_plot(data, ax, 2, "#BF5700")
        # batch_plot("line", data, "test", fig, ax,
        #        trace_color="#BF5700", xlabel="Wavelength (nm)", ylabel="Transmission", xrange=[np.min(wl), np.max(wl), 2.3], yrange=[0, 1.001, 0.2], barwidth=0.1/6, linewidth=2, fontsize=14, figsize_pixels=[400, 300], smoothness=0)
    plt.savefig(f"{dirname}/MDCurve2.png")




if __name__ == "__main__":
    # test_voltage_quantization()
    # plot_phase_voltage_curve()
    plot_md_curve()
