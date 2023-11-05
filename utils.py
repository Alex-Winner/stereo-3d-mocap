import numpy as np
from scipy import linalg


def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
    return P

def triangulate_point(P1, P2, point1, point2):
    """
    Triangulate a 3D point from two camera projection matrices and corresponding image coordinates.

    Parameters:
        P1 (numpy.ndarray): The 3x4 projection matrix for the first camera.
        P2 (numpy.ndarray): The 3x4 projection matrix for the second camera.
        point1 (numpy.ndarray): Pixel coordinates (x, y) of a feature point in the first image.
        point2 (numpy.ndarray): Pixel coordinates (x, y) of the same feature point in the second image.

    Returns:
        numpy.ndarray: The triangulated 3D coordinates of the feature point in world coordinates.

    This function takes two projection matrices (P1 and P2) and their corresponding image coordinates (point1 and point2).
    It calculates the 3D coordinates of a feature point in the world by triangulating the projections from both cameras.

    The projection matrices P1 and P2 should be 3x4 matrices that represent the camera's projection transformation.
    The input points point1 and point2 are 2D pixel coordinates specifying the location of the feature in the images.
    The returned 3D point is represented as a numpy array (x, y, z) in world coordinates.
    """
    # Construct matrix A for linear system of equations
    A = [
        point1[1] * P1[2, :] - P1[1, :],
        P1[0, :] - point1[0] * P1[2, :],
        point2[1] * P2[2, :] - P2[1, :],
        P2[0, :] - point2[0] * P2[2, :]
    ]
    A = np.array(A).reshape((4, 4))

    # Calculate matrix B from A
    B = A.transpose() @ A

    # Perform Singular Value Decomposition (SVD) on matrix B
    U, s, Vh = linalg.svd(B, full_matrices=False)

    # Triangulate the 3D point by dividing the last row of Vh by its last element
    triangulated_point = Vh[3, 0:3] / Vh[3, 3]

    return triangulated_point


def read_camera_parameters(camera_id):

    inf = open('camera_parameters/c' + str(camera_id) + '.dat', 'r')

    cmtx = []
    dist = []

    line = inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        cmtx.append(line)

    line = inf.readline()
    line = inf.readline().split()
    line = [float(en) for en in line]
    dist.append(line)

    return np.array(cmtx), np.array(dist)

def read_rotation_translation(camera_id, savefolder = 'camera_parameters/'):

    inf = open(savefolder + 'rot_trans_c'+ str(camera_id) + '.dat', 'r')

    inf.readline()
    rot = []
    trans = []
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        rot.append(line)

    inf.readline()
    for _ in range(3):
        line = inf.readline().split()
        line = [float(en) for en in line]
        trans.append(line)

    inf.close()
    return np.array(rot), np.array(trans)

def _convert_to_homogeneous(pts):
    pts = np.array(pts)
    if len(pts.shape) > 1:
        w = np.ones((pts.shape[0], 1))
        return np.concatenate([pts, w], axis = 1)
    else:
        return np.concatenate([pts, [1]], axis = 0)

def get_projection_matrix(camera_id):

    #read camera parameters
    cmtx, dist = read_camera_parameters(camera_id)
    rvec, tvec = read_rotation_translation(camera_id)

    #calculate projection matrix
    P = cmtx @ _make_homogeneous_rep_matrix(rvec, tvec)[:3,:]
    return P

def write_keypoints_to_disk(filename, kpts):
    fout = open(filename, 'w')

    for frame_kpts in kpts:
        for kpt in frame_kpts:
            if len(kpt) == 2:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ')
            else:
                fout.write(str(kpt[0]) + ' ' + str(kpt[1]) + ' ' + str(kpt[2]) + ' ')

        fout.write('\n')
    fout.close()

if __name__ == '__main__':

    P2 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)
