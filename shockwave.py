import numpy as np

def bellig(x, Mach):
    """
    Bellig's shockwave equation

    Parameters:
    x : array-like
        x coordinates
    Mach : float
        Mach number

    Returns:
    y : array-like
        y coordinates
    """
    beta = 0.4
    R = 3.38
    Rc = R * 1.143 * np.exp(0.54 / (Mach - 1)**1.2)
    y = Rc / np.tan(beta) * (((x) / (Rc * np.tan(beta)**(-2)) + 1)**2 - 1)**0.5
    return y

def swequations(gamma, M, beta):
    """
    Shockwave equations

    Parameters:
    gamma : float
        Specific heat relation for the gas
    M : float
        Mach number
    beta : float
        Shockwave angle

    Returns:
    M2 : float
        After shockwave Mach number
    p2p1 : float
        Static pressure relation
    rho2rho1 : float
        Static density relation
    T2T1 : float
        Static temperature relation
    p02p01 : float
        Total pressure relation
    deflangle : float
        Deflection angle of the flow
    """
    Mn = M * np.sin(beta)

    deflangle = np.arctan(2 * (np.tan(beta) * (M**2 * np.sin(beta)**2 - 1) / (M**2 * (gamma + np.cos(2 * beta)) + 2)))

    T2T1 = (2 * gamma * M**2 * np.sin(beta)**2 - (gamma - 1)) * ((gamma - 1) * M**2 * np.sin(beta)**2 + 2) / ((gamma + 1)**2 * M**2 * np.sin(beta)**2)
    p2p1 = (2 * gamma * M**2 * np.sin(beta)**2 - (gamma - 1)) / (gamma + 1)
    rho2rho1 = (gamma + 1) * M**2 * np.sin(beta)**2 / ((gamma - 1) * M**2 * np.sin(beta)**2 + 2)
    M2 = np.sqrt(1 / np.sin(beta - deflangle)**2 * ((gamma - 1) * M**2 * np.sin(beta)**2 + 2) / (2 * gamma * M**2 * np.sin(beta)**2 - (gamma - 1)))

    p02p01 = ((1 + (gamma - 1) / 2 * Mn**2) / (1 + (gamma - 1) / 2 * M2**2))**(gamma / (gamma - 1)) * (p2p1)**(-1 / (gamma - 1))

    return M2, p2p1, rho2rho1, T2T1, p02p01, deflangle