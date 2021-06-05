import os, sys
sys.path.insert(0, './submodules/packnet-sfm')

import torch
from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.datasets.augmentations import resize_image, to_tensor
from packnet_sfm.utils.depth import write_depth, inv2depth, viz_inv_depth
from packnet_sfm.utils.config import parse_test_file
from packnet_sfm.geometry.pose_utils import pose_vec2mat

class DepthPoseEstimator(object):
    def __init__(self):
        model_name = "semi_sup.ckpt"
        self.config, state_dict = parse_test_file(model_name)
        self.model_wrapper = ModelWrapper(self.config, load_datasets=False)
        self.model_wrapper.load_state_dict(state_dict)

        if torch.cuda.is_available():
            self.model_wrapper = self.model_wrapper.to("cuda")

        self.model_wrapper.eval()
        self.image_shape = self.config.datasets.augmentation.image_shape

    def get_depth(self, image):
        image = self.process_pil_image(image)

        pred_inv_depth = self.model_wrapper.depth(image)[0]
        depth = inv2depth(pred_inv_depth).squeeze()

        return depth.detach().numpy()

    def get_pose(self, cur_image, prev_image_1, prev_image_2):
        cur_image = self.process_pil_image(cur_image)
        prev_image_1 = self.process_pil_image(prev_image_1)
        prev_image_2 = self.process_pil_image(prev_image_2)

        pose = self.model_wrapper.pose(prev_image_1, (cur_image, prev_image_1))
        T = pose_vec2mat(pose[0])

        return T[0].detach().numpy()

    def process_pil_image(self, image):
        image = resize_image(image, self.image_shape)
        image = to_tensor(image).unsqueeze(0)

        if torch.cuda.is_available():
            image = image.to("cuda")

        return image