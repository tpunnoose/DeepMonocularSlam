import numpy as np
import cv2 as cv

class ImageFrame(object):
    def __init__(self, image, camera):
        self.image = image
        self.camera = camera

        self.detector = cv.ORB_create()
        self.extractor = self.detector

    def get_features(self):
        self.keypoints = self.detector.detect(self.image)
        self.descriptors = self.extractor.compute(self.image, self.keypoints)[1]
        self.matched_features = np.zeros(len(self.keypoints), dtype=bool)

    def display_keypoints(self, delay = 1):
        assert not self.keypoints == None 
        img = cv.drawKeypoints(self.image, self.keypoints, None, flags=0)
        cv.imshow("Keypoints Left", img)
        cv.waitKey(delay)


class StereoFrame(object):
    def __init__(self, frame_left, frame_right):
        self.left = frame_left
        self.right = frame_right
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    def match_stereo_features(self, matching_distance=40, max_row_distance=2.5, max_disparity=100):
        '''
        Use self.matcher to return matches between two stereo frames. 
        '''
        all_matches = self.matcher.match(self.left.descriptors, self.right.descriptors)
        self.matches = []
        
        for match in all_matches:
            left_pt = self.left.keypoints[match.queryIdx].pt
            right_pt = self.right.keypoints[match.trainIdx].pt

            if match.distance < matching_distance and \
                abs(left_pt[1] - right_pt[1]) < max_row_distance and \
                abs(left_pt[0] - right_pt[0]) < max_disparity:
                self.matches.append(match)
                self.left.matched_features[match.queryIdx] = True
                self.right.matched_features[match.trainIdx] = True

        # print("Number of successful stereo matches: ", len(self.matches))

    def triangulate(self):
        left_points = np.array([self.left.keypoints[m.queryIdx].pt for m in self.matches])
        right_points = np.array([self.right.keypoints[m.trainIdx].pt for m in self.matches])
        three_d_points = cv.triangulatePoints(self.left.camera, self.right.camera, left_points.T, right_points.T).T

        self.point_dict_left = dict(zip([m.queryIdx for m in self.matches], list(three_d_points)))
        
