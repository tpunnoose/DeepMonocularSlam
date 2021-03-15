from frame import StereoFrame, ImageFrame
from pose import SE3Pose, estimate_pose, pose_error
import numpy as np
import cv2 as cv
import pykitti

class StereoOdometry(object):
    def __init__(self, dataset, verbose=False):
        self.dataset = dataset
        self.dataset_iterator = iter(dataset.gray)
        self.current_pose = SE3Pose(np.eye(3), np.zeros(3)) 
        self.pose_history = [self.current_pose]
        self.initialized = False

        self.left_camera = dataset.calib.P_rect_00
        self.right_camera = dataset.calib.P_rect_10

        self.verbose = verbose

        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    def process_next_img(self, visualize=False):
        image_pair = next(self.dataset_iterator)

        left_frame = ImageFrame(np.array(image_pair[0]), self.left_camera)
        left_frame.get_features()

        if visualize:
            left_frame.display_keypoints()

        right_frame = ImageFrame(np.array(image_pair[1]), self.right_camera)
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
        

    def temporal_matches(self, prev_stereo_frame, next_stereo_frame, matching_distance=50):
        all_matches = self.matcher.match(prev_stereo_frame.left.descriptors, next_stereo_frame.left.descriptors)

        matches = []

        prev_image_points = []
        prev_3d_points = []

        next_image_points = []
        next_3d_points = []

        for match in all_matches:
            if match.distance < matching_distance and \
                prev_stereo_frame.left.matched_features[match.queryIdx] and \
                next_stereo_frame.left.matched_features[match.trainIdx]:
                matches.append(match)

                prev_image_points.append(prev_stereo_frame.left.keypoints[match.queryIdx].pt)
                prev_3d_points.append(prev_stereo_frame.point_dict_left[match.queryIdx])

                next_image_points.append(next_stereo_frame.left.keypoints[match.trainIdx].pt)
                next_3d_points.append(next_stereo_frame.point_dict_left[match.trainIdx])

        if self.verbose:
            print("Number of successful temporal matches: ", len(matches))

        return (np.array(prev_image_points), np.array(prev_3d_points), np.array(next_image_points), np.array(next_3d_points))

if __name__ == '__main__':
    basedir = '/Users/tarun/Classes/CS231A/project/KITTI/odometry/dataset'
    sequence = '04'
    dataset = pykitti.odometry(basedir, sequence)

    so = StereoOdometry(dataset)

    num_frames = min(300, len(dataset))
    for i in range(num_frames):
        so.process_next_img(True)
        true_pose = SE3Pose(dataset.poses[i][0:3, 0:3], dataset.poses[i][0:3, 3])

        r_e, t_e = pose_error(so.current_pose, true_pose)

        print("Current t: ", so.current_pose.t)
        print("True t: ", true_pose.t)


