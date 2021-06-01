from frame import StereoFrame, ImageFrame
from pose import SE3Pose, estimate_pose, pose_error
import numpy as np
import cv2 as cv
import pykitti

import sys
sys.path.insert(0, './disk')

from disk_feature import DiskFeature2D

class StereoOdometry(object):
    def __init__(self, dataset, verbose=False):
        self.dataset = dataset
        self.image_iterator = iter(dataset.gray)
        self.true_pose_iterator = iter(dataset.poses)
        self.current_pose = SE3Pose(np.eye(3), np.zeros(3)) 
        self.pose_history = [self.current_pose]
        self.initialized = False

        self.trajectory_image = np.zeros((1200, 1200, 3), dtype=np.int32)

        self.left_camera = dataset.calib.P_rect_00
        self.right_camera = dataset.calib.P_rect_10

        self.verbose = verbose
        self.detector = DiskFeature2D()

        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        # self.matcher = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

    def process_next_img(self, visualize=False):
        image_pair = next(self.image_iterator)
        self.true_pose = next(self.true_pose_iterator)
        self.true_pose = SE3Pose(self.true_pose[0:3, 0:3], self.true_pose[0:3, 3])

        left_frame = ImageFrame(np.array(image_pair[0].convert("RGB")), self.left_camera, self.detector)
        left_frame.get_features()

        right_frame = ImageFrame(np.array(image_pair[1].convert("RGB")), self.right_camera, self.detector)
        right_frame.get_features()

        self.next_stereo_frame = StereoFrame(left_frame, right_frame)
        self.next_stereo_frame.match_stereo_features()
        self.next_stereo_frame.triangulate()

        if self.initialized:
            prev_image_points, prev_3d_points, next_image_points, next_3d_points = self.temporal_matches(self.prev_stereo_frame, self.next_stereo_frame)
            pose_guess = SE3Pose(np.zeros(3), np.zeros(3))
            next_pose = estimate_pose(prev_image_points, prev_3d_points, next_image_points, next_3d_points, self.left_camera)
            # next_pose.R = np.eye(3)
            self.current_pose = self.current_pose.compose(next_pose)
            self.pose_history.append(self.current_pose)
        else:
            self.initialized = True
            
        self.prev_stereo_frame = self.next_stereo_frame

        if visualize:
            left_frame.display_keypoints()
            self.display_trajectory()

    def display_trajectory(self, offset=np.array([500, 500])):
        estimated_coordinates = (self.current_pose.t[[0, 2]] + offset).astype(np.int32)
        true_coordinates = (self.true_pose.t[[0, 2]] + offset).astype(np.int32)

        self.trajectory_image = cv.circle(self.trajectory_image, tuple(estimated_coordinates), 1, (0,0,255), 2)
        self.trajectory_image = cv.circle(self.trajectory_image, tuple(true_coordinates), 1, (0,255,0), 2)

        cv.imshow('Trajectory', self.trajectory_image.astype(np.uint8))
        cv.waitKey(1)
        

    def temporal_matches(self, prev_stereo_frame, next_stereo_frame, matching_distance=50):
        all_matches = self.matcher.match(prev_stereo_frame.left.descriptors, next_stereo_frame.left.descriptors)

        self.matches = []

        prev_image_points = []
        prev_3d_points = []

        next_image_points = []
        next_3d_points = []

        for match in all_matches:
            if match.distance < matching_distance and \
                prev_stereo_frame.left.matched_features[match.queryIdx] and \
                next_stereo_frame.left.matched_features[match.trainIdx]:
                self.matches.append(match)

                prev_image_points.append(prev_stereo_frame.left.keypoints[match.queryIdx].pt)
                prev_3d_points.append(prev_stereo_frame.point_dict_left[match.queryIdx])

                next_image_points.append(next_stereo_frame.left.keypoints[match.trainIdx].pt)
                next_3d_points.append(next_stereo_frame.point_dict_left[match.trainIdx])

        if self.verbose:
            print("Number of successful temporal matches: ", len(matches))

        return (np.array(prev_image_points), np.array(prev_3d_points), np.array(next_image_points), np.array(next_3d_points))

if __name__ == '__main__':
    basedir = '/Users/tarun/Classes/CS231A/project/KITTI/odometry/dataset'
    sequence = '00'
    dataset = pykitti.odometry(basedir, sequence)

    so = StereoOdometry(dataset)

    num_frames = min(len(dataset), 1000)
    for i in range(num_frames):
        so.process_next_img(True)
        r_e, t_e = pose_error(so.current_pose, so.true_pose)

        print("Current t: ", r_e)
        print("True t: ", t_e)

    cv.imwrite('trajectory' + sequence + '.png', so.trajectory_image.astype(np.uint8))



