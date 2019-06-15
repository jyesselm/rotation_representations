lerimport numpy as np
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

print

exit()

for i in range(100):
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
print euler
print R_new
