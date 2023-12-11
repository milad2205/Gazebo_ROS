import math


def circular_inter_fast(x1, z1, r1, x2, z2, r2, dir):
    alpha = -2 * (x1 - x2)
    beta = -2 * (z1 - z2)
    gamma = r1**2 - r2**2 - (z1**2 - z2**2) - (x1**2 - x2**2)

    A = alpha**2 + beta**2
    B = 2 * alpha * beta * x1 - 2 * alpha**2 * z1 - 2 * gamma * beta
    C = gamma**2 - 2 * alpha * x1 * gamma + alpha**2 * (x1**2 + z1**2 - r1**2)
    out_range = 0

    if B**2 - 4 * A * C < 0.0:
        out_range = 1
        zf = 0.0
        xf = 0.0
    else:
        zf1 = (-B + math.sqrt(B**2 - 4 * A * C)) / (2 * A)
        xf1 = (gamma - beta * zf1) / (alpha + 1e-10)
        zf2 = (-B - math.sqrt(B**2 - 4 * A * C)) / (2 * A)
        xf2 = (gamma - beta * zf2) / (alpha + 1e-10)
        if dir == 1:
            if xf1 > xf2:
                xf = xf1
                zf = zf1
            else:
                xf = xf2
                zf = zf2
        else:
            if xf1 < xf2:
                xf = xf1
                zf = zf1
            else:
                xf = xf2
                zf = zf2
    return xf, zf, out_range
