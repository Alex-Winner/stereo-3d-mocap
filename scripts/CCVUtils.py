# ---- Import ----
# Math
import numpy as np
import math

class CCVUtils:
    @staticmethod
    def rotation_matrix_from_vectors(vec1, vec2):                   # rotation matrix that aligns vec1 to vec2
        
        # Find the rotation matrix that aligns vec1 to vec2
        # :param vec1: A 3d "source" vector
        # :param vec2: A 3d "destination" vector
        # :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)

        k_mat = np.array([[0,     -v[2],   v[1]], 
                          [v[2],   0,     -v[0]], 
                          [-v[1],  v[0],   0]])

        rotation_matrix = np.eye(3) + k_mat + k_mat.dot(k_mat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    @staticmethod
    def vector_angle(u, v):                                         # angle between two vectors 'u' and 'v'
        
        # Calculation of angle between two vectors 'u' and 'v'
        # :param u: vector represented by [x, y, z]
        # :param v: vector represented by [x, y, z]
        # :return angle: angle between two vectors in radian
        
        angle = np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return angle

    @staticmethod
    def RadToDeg(angleRad):                                         # Radian to Degree
        return math.degrees(angleRad)

    @staticmethod
    def DegToRad(angleDeg):                                         # Degree to Radian 
        return math.radians(angleDeg)

    @staticmethod
    def distance_planes_to_point(planes, point, absolute=True):     # Find distances to all planes
        
        x, y, z = point
        a = planes[:, 0]
        b = planes[:, 1]
        c = planes[:, 2]
        d = planes[:, 3]

        if absolute:
            distances = np.abs(a * x + b * y + c * z + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        else:
            distances = (a * x + b * y + c * z + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        return distances

    @staticmethod
    def DistancePlaneToPoint(plane, point):                         # Find distance from plane to point
        a, b, c, d = plane
        x, y, z = point

        distance = (a * x + b * y + c * z + d) / np.sqrt(a ** 2 + b ** 2 + c ** 2)
        return distance

    @staticmethod
    def translation_mat(tx, ty, tz):                                # Translation matrix
        return np.array([[1, 0, 0, tx], 
                         [0, 1, 0, ty], 
                         [0, 0, 1, tz], 
                         [0, 0, 0, 1]])

    @staticmethod
    def scaling_mat(sx, sy, sz):                                    # Scaling matrix
        return np.array([[sx, 0, 0, 0], 
                         [0, sy, 0, 0], 
                         [0, 0, sz, 0], 
                         [0, 0, 0, 1]])

    @staticmethod
    def rot_x_mat(theta):                                           # Rotate X
        return np.array([[1, 0,                 0,              0],
                         [0, np.cos(theta),     np.sin(theta),  0],
                         [0, -np.sin(theta),    np.cos(theta),  0],
                         [0, 0,                 0,              1]])

    @staticmethod
    def rot_y_mat(theta):                                           # Rotate Y
        return np.array([[np.cos(theta),    0, -np.sin(theta),  0],
                         [0,                1, 0,               0],
                         [np.sin(theta),    0, np.cos(theta),   0],
                         [0,                0, 0,               1]])

    @staticmethod
    def rot_z_mat(theta):                                           # Rotate Z
        return np.array([[np.cos(theta),    -np.sin(theta), 0, 0],
                         [np.sin(theta),    np.cos(theta),  0, 0],
                         [0,                0,              1, 0],
                         [0,                0,              0, 1]])