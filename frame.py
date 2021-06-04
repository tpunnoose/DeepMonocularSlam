import numpy as np
import cv2 as cv

class ImageFrame(object):
    def __init__(self, image, image_pil, camera, detector):
        self.image = image
        self.camera = camera

        # self.detector = cv.ORB_create()
        self.detector = detector
        self.image_pil = image_pil

    def get_features(self):
        self.keypoints, self.descriptors = self.detector.detectAndCompute(self.image, None)
        self.matched_features = np.zeros(len(self.keypoints), dtype=bool)

    def display_keypoints(self, delay = 1):
        assert not self.keypoints == None 
        img = cv.drawKeypoints(self.image, self.keypoints, None, flags=0)
        cv.imshow("Keypoints Left", img)
        cv.waitKey(delay)

    def process_depth(self):
        for kp in self.keypoints:
            point_3d = self.depth[kp.pt[0], kp.pt[1]] * np.linalg.inv(self.camera[0:3, 0:3]) @ kp.pt