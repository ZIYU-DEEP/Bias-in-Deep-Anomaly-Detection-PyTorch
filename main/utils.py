import numpy as np
from math import sin, cos


def gen_ball(r, mu, step=30):
    """
    Generate a set of data points surrounding a point like a ball.
    This function will later be used in main to generate a set of means
    for abnormal test.

    Inputs:
        r: (float) distance between the trained normal and the trained
           abnormal; used as the radius here
        mu: (np.array) a 3d array specifying the mu for trained normal
            or the trained abnormal data
        step: (int) the step (degree / du) to get the samples

    Returns:
        result: (list) a list a 3d arrays indicating the mean for abnormal
                data to test
    """
    thetas = range(0, 360, step)
    phis = range(0, 360, step)
    pairs = [(theta, phi) for theta in thetas for phi in phis]

    result = []
    for pair in pairs:
        theta, phi = pair
        cord = [sin(theta) * cos(phi) * r + mu[0],
                sin(theta) * sin(phi) * r + mu[1],
                cos(theta) * r + mu[2]]
        if cord in result:
            continue
        result.append(cord)

    return result
