"""
Filename: utils.py
Usage: addtional helper functions for the network.
"""

from math import sin, cos

import numpy as np
import logging





# #############################################################
# General Helper Functions
# #############################################################
def set_logger(log_path):
    """
    Set up a logger for use.
    """
    # Config the logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    # Set level and formats
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Set the file handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Record logging
    logger.info(log_path)

    return logger


def str_to_list(arg: str = '10-7-5-4-3'):
    '''
    Turn a string of numbers into a list.

    Inputs:
      arg: (str) must be of the form like "1-2-3-4".

    Returns:
      (list) example: [1, 2, 3, 4].
    '''

    return [int(i) for i in arg.strip().split('-')]


# #############################################################
# Misc
# #############################################################
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
