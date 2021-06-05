import numpy as np
import cv2 as cv

class ImageFrame(object):
    def __init__(self, image, image_pil, camera, detector):
        self.image = image
        self.camera = camera
        self.camera_inv = np.linalg.inv(camera[0:3, 0:3])

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
        self.points_3d = np.zeros((3, len(self.keypoints)))
        for i, kp in enumerate(self.keypoints):
            self.points_3d[:, i] = self.depth[int(kp.pt[1]), int(kp.pt[0])] * self.camera_inv @ np.hstack((kp.pt, 1))

        self.points_3d = np.vstack((self.points_3d, np.ones(len(self.keypoints))))
