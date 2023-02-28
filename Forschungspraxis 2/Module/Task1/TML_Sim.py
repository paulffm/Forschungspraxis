import gmsh
import numpy as np
from matplotlib import pyplot as plt
import analytic_sol



def A(L, C, R, f, depth):
    """The propagation matrix A."""
    w = 2 * np.pi * f
    Z_i = (R + (w * L) * 1j)
    Y_i = (w * C) * 1j
    beta = np.sqrt(Z_i * Y_i)
    bl = beta * depth
    Z_char = np.sqrt(Z_i / Y_i)
    Y = 1 / Z_char
    return np.array([
        [np.cosh(bl), -Z_char * np.sinh(bl)],
        [-Y * np.sinh(bl), np.cosh(bl)]
    ])

def B(L, C, R, f, depth):
    """The admittance matrix B."""
    w = 2 * np.pi * f
    Z_i = (R + (w * L) * 1j)
    Y_i = (w * C) * 1j
    beta = np.sqrt(Z_i * Y_i)
    bl = beta * depth
    Y = (w * C) * 1j
    return np.array([
        [np.cosh(bl), -1],
        [1, -np.cosh(bl)]
    ]) * Y / np.sinh(bl)

# Fragen: Plot für Phase korrekt? Interpretation: wenn shifting von -
def main():
    # constant
    r_1 = 2e-3  # [m]: inner radius (wire)
    r_2 = 3.5e-3  # [m]: outer radius (shell)
    depth = 300e-3  # [m]: Depth of wire (lz)
    model_name = "wire"  # str : model name
    I = 16  # [A]   : applied current
    J_0 = I / (np.pi * r_1 ** 2)  # [A/m] : current density in the wire
    mu_0 = 4 * np.pi * 1e-7  # [H/m] : permeability of vacuum (and inside of wire)
    mu_shell = 5 * mu_0
    eps_0 = 8.8541878128 * 1e-12

    ##### Task 10: TLM #####
    sigma = 57.7e6

    # inductivity per length
    L = analytic_sol.Inductance() / depth
    print('L', L)

    # conductivity per length
    C = (2 * np.pi * eps_0) / np.log(r_2 / r_1)
    print('C', C)
    # wrong
    # print('C_num', eps_0 * mu_shell / L)

    # resistance per length
    R = 1 / (sigma * np.pi * r_1 ** 2)

    # impedance matrix
    Z = []
    Y = []
    w = []
    beta = []
    Z_char = []
    c_0 = 1 / np.sqrt(mu_0 * eps_0)
    v_phase = c_0 #1 / (np.sqrt(L * C))
    print(v_phase, c_0)

    wave_len = []
    #f = np.arange(1, 10e3 + 1, 1)
    f = np.logspace(0, 5, 100)

    for i in range(len(f)):
        w_i = 2 * np.pi * f[i]
        w.append(w_i)
        # impedance matrix
        Z_i = (R + (w_i * L) * 1j)
        Z.append(Z_i)

        Y_i = (w_i * C) * 1j
        Y.append(Y_i)
        beta.append(np.sqrt(Z_i * Y_i))
        Z_char.append(np.sqrt(Z_i / Y_i))
        wave_len.append(v_phase / f[i])

    Z_abs = np.abs(Z_char)
    Z_ang = np.angle(Z_char, deg=True)


    plt.xlabel('frequency')
    plt.ylabel('abs(Z)')
    plt.title('Bode plot of abs(Z_char) in loglog')
    plt.loglog(f, Z_abs)
    plt.show()


    plt.xlabel('frequency')
    plt.ylabel('angle(Z) in Degrees')
    plt.title('Plot of angle(Z_char) in semilog')
    plt.plot(f, Z_ang)
    plt.xscale('log')
    plt.show()

    plt.xlabel('frequency')
    plt.ylabel('wave length')
    plt.title('Plot of wave length in semilog')
    plt.plot(f, wave_len)
    plt.xscale('log')
    plt.show()
    print(wave_len)

    plt.xlabel('frequency')
    plt.ylabel('v_phase')
    plt.title('plot of v_phase in semilog')
    plt.plot(f, wave_len * f)
    plt.xscale('log')
    plt.xlabel('frequency')
    plt.ylabel('v_phase')
    plt.show()

    # propagation and admittance matrix for f_3 = 1kHz
    f_3 = 1e3
    prop_mat = A(L, C, R, f_3, depth)
    print('propagation matrix', prop_mat)
    adm_mat = B(L, C, R, f_3, depth)
    print('Admittance matrix', adm_mat)

if __name__ == '__main__':
    main()