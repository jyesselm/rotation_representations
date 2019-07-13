import numpy as np
import pandas as pd
import math
import rnamake.transformations as t
from rnamake import util

def rotation_matrix_to_euler_angles(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def euler_flip(e):
    new_e = np.array(e)
    for i in range(3):
        if e[i] > 0:
            new_e[i] -= math.pi
        else:
            new_e[i] += math.pi

    new_e[1] = -new_e[1]
    return new_e

def euler_sum(e):
    sum = abs(e[0]) + abs(e[1]) + abs(e[2])
    return sum

def norm_euler_angles(e):
    new_e = np.array(e)
    new_e_2 = euler_flip(new_e)
    sum_e = euler_sum(new_e)
    sum_e_2 = euler_sum(new_e_2)
    #print new_e, new_e_2
    if sum_e_2 < sum_e:
        new_e = new_e_2
    if new_e[0] < 0:
        new_e = -new_e
    return new_e

def rotation_matrix_to_norm_euler_angles(R):
    e_angles = rotation_matrix_to_euler_angles(R)
    return norm_euler_angles(e_angles)



def euler_angles_to_rotation_matrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

"""for i in range(100):
    r = t.random_rotation_matrix()[:3, :3]
    euler = rotationMatrixToEulerAngles(r)
    r_new = eulerAnglesToRotationMatrix(euler)

    if util.matrix_distance(r, r_new) > 0.001:
        print "made it"


exit()


R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
euler = rotationMatrixToEulerAngles(R)

R_new = eulerAnglesToRotationMatrix(euler)

R = np.array([[0.411982, -0.833738, -0.367630],
              [-0.058727, -0.426918, 0.902382],
              [-0.909297, -0.350175, -0.224845]])

euler = rotationMatrixToEulerAngles(R)
R_new = eulerAnglesToRotationMatrix(euler)

print R
print euler"""
