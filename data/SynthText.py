import scipy.io as scio
import os
# import torch
# import torch.utils.data as data
import cv2
import numpy as np
import re
import itertools

from gaussianMask.gaussian import GaussianTransformer
from data.boxEnlarge import enlargebox


class craftDataset(object):
    def __init__(self, target_size=768, data_dir_list={"synthtext":"datapath"}, vis=False):
        assert 'synthtext' in data_dir_list.keys()
        
        self.target_size = target_size
        self.data_dir_list = data_dir_list
        self.vis = vis

        self.charbox, self.image, self.imgtxt = self.load_synthtext()

    def load_synthtext(self):
        gt = "/home/neeraj/Desktop/IFT6759/CRAFT/DS/SynthText/gt.mat"
        #gt = scio.loadmat(os.path.join(self.data_dir_list["synthtext"], 'gt.mat'))
        charbox = gt['charBB'][0]
        image = gt['imnames'][0]
        imgtxt = gt['txt'][0]
        return charbox, image, imgtxt

    def load_synthtext_image_gt(self, index):
        img_path = os.path.join(self.data_dir_list["synthtext"], self.image[index][0])
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        _charbox = self.charbox[index].transpose((2, 1, 0))
        # image = random_scale(image, _charbox, self.target_size)

        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]
        character_bboxes = []
        total = 0
        for i in range(len(words)):
            bboxes = _charbox[total:total + len(words[i])]
            assert (len(bboxes) == len(words[i]))
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)

        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), img_path