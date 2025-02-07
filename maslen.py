import numpy as np
import matplotlib.pyplot as plt

from shockwave import bellig, swequations

def maslen():
    # Maslen method function
    ns = 100  # number of points per streamline
    nj = 100  # number of points perpendicular to streamline

    gamma = 1.4
    Mach = 25.6
    w = 0  # 2D flow

    # initialize arrays:
    u = np.zeros((ns, nj))
    p = np.zeros((ns, nj))
    rho = np.zeros((ns, nj))
    psi = np.zeros((ns, nj))
    y = np.zeros((ns, nj))
    y2 = np.zeros((ns, nj))
    x2 = np.zeros((ns, nj))
    beta1 = np.zeros((ns, nj))
    p1 = np.zeros((ns, nj))
    rho1 = np.zeros((ns, nj))

    # x coordinate discretization with density function to have more points near the edge.
    x_min = 0
    x_max = 2.5
    density_factor = 5

    t = np.linspace(0, 1, nj)  # Uniform parameterization
    xs = x_min + (x_max - x_min) * (t ** density_factor)  # Exponential scaling
    ys = bellig(xs, Mach)

    # shockwave geometry parameters
    beta = slope(xs, Mach)
    k = curvature(xs, Mach)

    u[:, 0] = np.cos(beta)  # dimensionless velocity after shockwave

    # after shockwave conditions
    _, p2p1, rho2rho1, _, _, _ = swequations(gamma, Mach, beta)

    # after shockwave conditions
    rho[:, 0] = rho2rho1  # dimensionless density
    p[:, 0] = p2p1 / (gamma * Mach ** 2)  # dimensionless pressure
    psi[:, 0] = ys  # dimensionless streamline

    # set boundary values
    y[:, 0] = 0
    x2[:, 0] = xs
    y2[:, 0] = ys

    # breakpoint()

    for i in range(ns):
        for j in range(1, nj):
            if j == nj - 1:
                # streamline at the body
                psi[i, j] = 0
            else:
                # local streamline
                psi[i, j] = psi[i, 0] - (j - 1) * (psi[i, 0] / (nj - 1))
            # pressure calculation at each point
            p[i, j] = p[i, j - 1] + k[i] * u[i, 0] / ys[i] ** w * (psi[i, j] - psi[i, j - 1])

            # density calculation
            x1 = psi[i, j]

            beta1[i, 0] = beta[i]
            beta1[i, j] = slope(x1, Mach)

            sp1 = (2 * gamma) * ((Mach ** 2) * (np.sin(beta1[i, j])) ** 2 - 1) / (gamma + 1)

            p1[i, 0] = p[i, 0]
            p1[i, j] = (1 + sp1) / (gamma * Mach ** 2)

            rho1[i, 0] = rho[i, 0]
            rho1[i, j] = ((gamma + 1) * (Mach ** 2) * (np.sin(beta1[i, j])) ** 2) / (
                    ((gamma - 1) * (Mach ** 2) * (np.sin(beta1[i, j])) ** 2) + 2)

            # isentropic relations to calculate density
            rho[i, j] = rho1[i, j] * (p[i, j] / p1[i, j]) ** (1 / gamma)

            # enthalpy calculation
            h0 = (2 / ((gamma - 1) * Mach ** 2)) + 1  # stagnation enthalpy
            h = (2 * gamma / (gamma - 1)) * (p[i, j] / rho[i, j])  # local enthalpy

            # flow velocity at each point using enthalpy
            u[i, j] = ((h0 - h)) ** 0.5

            y[i, j] = y[i, j - 1] - (2 * (psi[i, j] - psi[i, j - 1])) / (rho[i, j] * u[i, j] + rho[i, j - 1] * u[i, j - 1])
            y2[i, j] = ((ys[i]) ** 2 - (2 * u[i, 0] * y[i, j])) ** 0.5

            if i == 0:
                x2[i, j] = np.nan
            else:
                x2[i, j] = xs[i] + (ys[i] - y2[i, j]) * np.tan(beta[i])
    
    plotter(xs, ys, x2, y2, p)


def slope(x, Mach):
    # beta angle calculation for Bellig's shockwave.
    b = 0.34
    R = 3.3897
    Rc = R * 1.143 * np.exp(0.54 / (Mach - 1) ** 1.2)

    beta = np.arctan(((x) / (Rc * np.tan(b) ** (-2)) + 1) / (np.tan(b) ** (-2) * np.tan(b) * (((x) / (Rc * np.tan(b) ** (-2)) + 1) ** 2 - 1) ** 0.5))

    return beta


def curvature(x, Mach):
    # curvature calculation for Bellig's shockwave.
    b = 0.34
    R = 3.3897
    Rc = R * 1.143 * np.exp(0.54 / (Mach - 1) ** 1.2)

    dydx = ((x) / (Rc * np.tan(b) ** (-2)) + 1) / (np.tan(b) ** (-2) * np.tan(b) * (((x) / (Rc * np.tan(b) ** (-2)) + 1) ** 2 - 1) ** 0.5)
    d2ydx2 = -np.tan(b) ** 3 / (Rc * (((np.tan(b) ** 2 * (x)) / Rc + 1) ** 2 - 1) ** 1.5)

    k = (abs(d2ydx2)) / ((1 + (dydx) ** 2) ** 1.5)

    return k


def plotter(xs, ys, x2, y2, p):
    plt.figure(1)
    plt.contourf(x2, y2, p, 100, cmap='jet')
    plt.colorbar()
    plt.plot(xs, ys, 'b-', linewidth=2, label='Shock Shape')
    plt.plot(x2[:, -1], y2[:, -1], 'r-', linewidth=2, label='Body')
    plt.xlabel('X Coordinate (m)')
    plt.ylabel('Y Coordinate (m)')
    plt.title('Pressure Profile with Shock Shape and Vehicle Body')
    plt.legend()
    plt.axis('equal')
    plt.show()



if __name__ == "__main__":
    maslen()