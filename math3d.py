import numpy as np
import math


def normalize(vector):
    ''' Return the input vector divided by its norm. The resulting vector is of magnitude 1
    The input vector must be a list. '''
    magn = magnitude(vector)
    v = np.asarray(vector, dtype=float).copy()
    v /= magn

    if isinstance(vector, np.ndarray):
        return v
    else:
        return v.tolist()


def magnitude(vector):
    sqr_sum = 0
    for val in vector:
        sqr_sum += val**2
    return math.sqrt(sqr_sum)


def cartesian2spherical(xyz):
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    radial = magnitude(xyz)
    azimuth = math.atan2(y, x)
    elevation = math.atan2(z, magnitude((x, y)))
    return [radial, azimuth, elevation]


def spherical2cartesian(rae):
    radial = rae[0]
    azimuth = rae[1]
    elevation = rae[2]

    x = radial*math.cos(azimuth)*math.cos(elevation)
    y = radial*math.sin(azimuth)*math.sin(elevation)
    z = radial*math.sin(elevation)
    return [x, y, z]


def axisAngle_to_Rmat(axis, angle):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    http://stackoverflow.com/a/6802723
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(angle/2.0)
    b, c, d = -axis*math.sin(angle/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([
        [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac), 0.0],
        [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab), 0.0],
        [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc, 0.0],
        [0.0, 0.0, 0.0, 1.0]
        ])


def clean_taitbryan_angle(thetas):
    """
    tait-bryan angle's space is not uniform and we must be careful when theta_y tends to +-pi/2.
    Indeed, when theta_y is +-pi/2, there is an infinite number of combination of (theta_x, theta_z)
    describing the same rotaton.
    In that case, we choose to set theta_z to 0, and compute theta_x accordingly.
    :param thetas: [theta_x, theta_y, theta_z]
    :return: corrected thetas: [theta_x, theta_y, theta_z]
    """
    thetas = thetas + [] # deep copy to avoid bad surprises
    R = thetas_to_Rmat(thetas)
    thetas = Rmat_to_thetas(R)

    pi = np.pi
    thetas[1] = min(pi, max(-pi, thetas[1]))  # clip theta_y in range [-pi/2, pi/2] because of rounding errors

    # most of the time we won't fall in the if case
    if pi / 2 - abs(thetas[1]) < 1e-6:
        thetas[0] -= math.copysign(thetas[2], thetas[1])
        thetas[2] = 0

    # force theta_x and theta_z to stay in range [-pi, pi)
    thetas[0] = (thetas[0] + pi) % (2 * pi) - pi
    thetas[2] = (thetas[2] + pi) % (2 * pi) - pi
    return thetas


def Rmat_to_axisAngle(R):
    #todo test this function
    """
    :param R: 
    :return: 
    """
    angle = math.acos(( R[0,0] + R[1,1] + R[2,2] - 1)/2)
    x = (R[2,1] - R[1,2]) / math.sqrt((R[2,1] - R[1,2])**2 + (R[0,2] - R[2,0])**2 + (R[1,0] - R[0,1])**2)
    y = (R[0,2] - R[2,0]) / math.sqrt((R[2,1] - R[1,2])**2 + (R[0,2] - R[2,0])**2 + (R[1,0] - R[0,1])**2)
    z = (R[1,0] - R[0,1]) / math.sqrt((R[2,1] - R[1,2])**2 + (R[0,2] - R[2,0])**2 + (R[1,0] - R[0,1])**2)
    axis = [x,y,z]
    return axis, angle


def thetas_to_Rmat(thetas, homogenous=True):
    """
    Compute the rotation matrix R, given the Tait-Bryan angle vector thetas = [theta_x, theta_y, theta_z]
    The rotation is applied in this order: R = Rz * Ry * Rx
    :param thetas: the 3 angles [theta_x, theta_y, theta_z] in radian 
    :param homogenous: if homogenous=True R will be (4x4), if homogenous=False R will be (3x3)
    :return: Rotation matrix R
    """

    cx = math.cos(thetas[0])
    sx = math.sin(thetas[0])
    cy = math.cos(thetas[1])
    sy = math.sin(thetas[1])
    cz = math.cos(thetas[2])
    sz = math.sin(thetas[2])

    Rx = np.matrix([[1.0, 0.0, 0.0],
                    [0.0, cx, -sx],
                    [0.0, sx, cx]])

    Ry = np.matrix([[ cy, 0.0,  sy],
                    [0.0, 1.0, 0.0],
                    [-sy, 0.0,  cy]])

    Rz = np.matrix([[ cz, -sz, 0.0],
                    [ sz,  cz, 0.0],
                    [0.0, 0.0, 1.0]])
    R = Rz * Ry * Rx

    if homogenous:
        R = np.hstack((R, np.array([[0], [0], [0]])))
        R = np.vstack((R, np.array([0,0,0,1])))
    return R


def quaternion_to_Rmat(qx, qy, qz, qw):
    #TODO test this function
    """
    found at: http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    :param qx: 
    :param qy: 
    :param qz: 
    :param qw: 
    :return: 
    """
    R = np.matrix([
        [1 - 2*qy**2 - 2*qz**2,	    2*qx*qy - 2*qz*qw,  	2*qx*qz + 2*qy*qw,  0],
        [    2*qx*qy + 2*qz*qw,	1 - 2*qx**2 - 2*qz**2, 	    2*qy*qz - 2*qx*qw,  0],
        [    2*qx*qz - 2*qy*qw,	    2*qy*qz + 2*qx*qw, 	1 - 2*qx**2 - 2*qy**2,  0],
        [                    0,                     0,                      0,  1]
    ])
    return R


def copysign0(a, b):
    """
    same as math.copysign but handle the 0 case
    """
    if b == 0:
        return 0
    else:
        return math.copysign(a, b)


def Rmat_to_quaternion(R):
    """
    code founded at: http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    :param R: 
    :return: 
    """
    # the use of max is just a safeguard against rounding error
    qw = math.sqrt(max(0, 1 + R[0,0] + R[1,1] + R[2,2])) / 2
    qx = math.sqrt(max(0, 1 + R[0,0] - R[1,1] - R[2,2])) / 2
    qy = math.sqrt(max(0, 1 - R[0,0] + R[1,1] - R[2,2])) / 2
    qz = math.sqrt(max(0, 1 - R[0,0] - R[1,1] + R[2,2])) / 2

    # copysign takes the sign from the second term and sets the sign of the first without altering the magnitude
    qx = copysign0(qx, R[2,1] - R[1,2])
    qy = copysign0(qy, R[0,2] - R[2,0])
    qz = copysign0(qz, R[1,0] - R[0,1])
    return qx, qy, qz, qw


def Rmat_to_thetas(R):
    """
    Extract Tait-Bryan angles from rotation matrix
    :param R: 
    :return: len-3 list of theta angles x, y, z in radian
    """
    singular_y = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    th_y = math.atan2(-R[2, 0], singular_y)  # similar to asin(-R[2, 0])
    singular = singular_y < 1e-6

    if not singular:
        # most of the time we fall in the non-singulare case
        th_x = math.atan2(R[2, 1], R[2, 2])
        th_z = math.atan2(R[1, 0], R[0, 0])
    else:
        # When th_y tends to pi/2 or -pi/2, then x and z rotation are mixed together leading to an infinity of solution.
        # So we set th_z to 0 to have a single solution
        th_x = math.atan2(-R[1, 2], R[1, 1])
        th_z = 0

    thetas = [th_x, th_y, th_z]
    return thetas


def thetas_to_quaternion(thetas):
    # TODO: enhance the doc
    """
    go through Rmat
    :param thetas: 
    :return: 
    """
    R = thetas_to_Rmat(thetas)
    qx, qy, qz, qw = Rmat_to_quaternion(R)
    return qx, qy, qz, qw


def quaternion_to_thetas(qx, qy, qz, qw):
    # TODO: write the doc for this taitbryan function
    R = quaternion_to_Rmat(qx, qy, qz, qw)
    thetas = Rmat_to_thetas(R)
    return thetas
