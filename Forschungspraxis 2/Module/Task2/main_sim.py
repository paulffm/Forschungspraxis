import numpy as np
import numpy.linalg as la
from matplotlib import pyplot as plt
from mTLM import Z, Y, decompose, Z_ch, beta_m, A_m, A, solve, P, Pout


def main():



    ## Task 1:
    r_w = 1.1e-3
    sigma = 57.7e6

    R_1 = np.eye(3) * (1 / (sigma * np.pi * r_w ** 2))
    print(R_1)


    f = 1e3
    l = 1 #2e3
    R = np.eye(3) * 1e-3
    G = np.zeros((3, 3))
    L = np.array([[5, 1, 1], [1, 5, 1], [1, 1, 5]]) * 1e-6
    C = np.array([[5, -1, -1], [-1, 5, -1], [-1, -1, 5]]) * 1e-9

    Z_z = Z(R, L, f)
    Y_y = Y(G, C, f)
    print('Z', Z_z, Z_z.shape)
    print('Y', Y_y, Y_y.shape)

    Zm, Ym, Tu, Ti = decompose(Z_z, Y_y)
    print('Zm', Zm, Zm.shape)
    print('Ym', Ym, Ym.shape)
    print('Tu', Tu, Tu.shape)
    print('Ti', Ti, Ti.shape)

    z_char = Z_ch(Zm, Ym)
    print('z_char', z_char, z_char.shape)
    b = beta_m(Zm, Ym)
    print('beta', b, b.shape)
    am = A_m(z_char, b, l)
    print('am', am, am.shape)
    a = A(am, Tu, Ti)
    print('a', a, a.shape)

    # Task 2 - Phase plot
    # i = Ti[:, 2]
    # X = [0, 0, 0]
    # re = [np.real(z) for z in i]
    # im = [np.imag(z) for z in i]
    # plt.quiver(X, X, re, im)
    # plt.show()

    # Task 3
    u0 = np.array([100, 80 * np.exp(np.pi * 2j / 3), 60 * np.exp(np.pi * 4j / 3)])
    r = 1
    u0, i0, ul, il = solve(a, u0, r)
    p_in, p_out = P(u0, i0, ul, il)
    print(f"Power loss {p_in - p_out}W and relative loss {100 * (p_in - p_out) / p_in}%")

    # Task 3 - Load power plot
    f = np.logspace(1, 6, 200)
    p_out = [Pout(fi, l, R, G, L, C, u0) for fi in f]
    # plt.plot(f, p_out)
    plt.loglog(f, p_out)
    # plt.semilogx(f, p_out)
    plt.show()



if __name__ == '__main__':
    main()
