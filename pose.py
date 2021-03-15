import numpy as np

from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

class SE3Pose(object):
    def __init__(self, R_mat, t, from_euler=False):
        '''
        Arguments:
            R_mat: rotation matrix or zxz Euler angle if from_euler=True
            t: transformation vector between frame origins
        '''
        if from_euler:
            self.R = R.from_euler('zxz', R_mat).as_matrix()
        else:
            self.R = R_mat
        self.t = t

    def inverse(self):
        return SE3Pose(self.R.T, -self.R.T @ self.t)

    def matrix(self):
        T = np.zeros((4,4))
        T[0:3, 0:3] = self.R
        T[0:3, 3] = self.t
        T[3,3] = 1

        return T

    def compose(self, pose2):
        '''
        Composes two SE3Pose objects and returns a new SE3Pose resulting object
        '''
        new_transform = self.matrix() @ pose2.matrix()
        R = new_transform[0:3, 0:3]
        t = new_transform[0:3, 3]

        return SE3Pose(R, t)
    

def estimate_pose(prev_image_points, prev_3d_points, next_image_points, next_3d_points, camera, initial_guess=np.zeros(6)):
    def reprojection(x):
        N = prev_image_points.shape[0]

        r = x[0:3]
        t = x[3:6]
        pose = SE3Pose(r, t, from_euler = True)
        prev_to_next = camera @ pose.inverse().matrix() @ prev_3d_points.T
        error_1 = next_image_points.T - prev_to_next[0:2, :]/prev_to_next[2, :]

        next_to_prev = camera @ pose.matrix() @ next_3d_points.T
        error_2 = prev_image_points.T - next_to_prev[0:2, :]/next_to_prev[2, :]

        return np.hstack([error_1.flatten(), error_2.flatten()])

    x0 = initial_guess
    result = least_squares(reprojection, x0)

    return SE3Pose(result.x[0:3], result.x[3:6], from_euler = True) 

def pose_error(pose1, pose2):
    t_e = np.linalg.norm(pose1.t - pose2.t)
    r_e = np.linalg.norm(R.from_matrix(pose1.R.T @ pose2.R).as_rotvec())

    return (r_e, t_e)
        
