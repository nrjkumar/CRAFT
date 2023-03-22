import scipy.io as scio
import os,sys
import torch,copy
import torch.utils.data as data
import torchvision.transforms as transforms
import cv2
import numpy as np
import re, random
import itertools
from PIL import Image
sys.path.append('/home/neeraj/Desktop/IFT6759/CRAFT')

from data import imgproc

from gaussianMask.gaussian import GaussianTransformer
from data.boxEnlarge import enlargebox
from data.imgaugment import random_scale, random_crop



def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

class SynthTextDataLoader(data.Dataset):
    def __init__(self, target_size=768, data_dir_list={"synthtext":"datapath"}, vis=False,debug=False):
          #def __init__(self, target_size=768, viz=False, debug=False):
        self.target_size = target_size
        self.viz = vis
        self.debug = debug
        self.gaussianTransformer = GaussianTransformer(imgSize=1024, region_threshold=0.35, affinity_threshold=0.15)

        assert 'synthtext' in data_dir_list.keys()

        # self.target_size = target_size
        self.data_dir_list = data_dir_list
        # self.vis = vis

        self.charbox, self.image, self.imgtxt = self.load_synthtext()
       # self.gaussianTransformer = GaussianTransformer(200, 1.5)
        #print(self.charbox)
        # self.gen.gen_circle_mask()
    def load_synthtext(self):
        gt = scio.loadmat(os.path.join(self.data_dir_list["synthtext"], 'gt.mat'))
        charbox = gt['charBB'][0]
        image = gt['imnames'][0]
        imgtxt = gt['txt'][0]
        return charbox, image, imgtxt

    def load_synthtext_image_gt(self, index):
        img_path = os.path.join(self.data_dir_list["synthtext"], self.image[index][0])
        #print(img_path)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        _charbox = copy.deepcopy(self.charbox[index].transpose((2, 1, 0)))
        image = random_scale(image, _charbox, self.target_size)
        words = [re.split(' \n|\n |\n| ', t.strip()) for t in self.imgtxt[index]]
        words = list(itertools.chain(*words))
        words = [t for t in words if len(t) > 0]

        character_bboxes = []
        total = 0
        confidences = []
        for i in range(len(words)):
            bboxes = _charbox[total:total + len(words[i])]
            assert (len(bboxes) == len(words[i]))
            total += len(words[i])
            bboxes = np.array(bboxes)
            character_bboxes.append(bboxes)
            confidences.append(1.0)

        return image, character_bboxes, words, np.ones((image.shape[0], image.shape[1]), np.float32), confidences, img_path

    def resizeGt(self, gtmask):
        return cv2.resize(gtmask, (self.target_size // 2, self.target_size // 2))

    def pull_item(self, index):
        image, character_bboxes, words, confidence_mask, confidences, img_path = self.load_synthtext_image_gt(index)
        sum = 0
        #print(confidence_mask)
       # print(character_bboxes)
        # region_scores = self.gen.generate_region(image.shape, character_bboxes)
        # affinities_scores, _ = self.gen.generate_affinity(image.shape, character_bboxes, words)
       # print(image)
        #print(character_bboxes)
        #print(region_scores)
       # print(affinities_scores)
       # print(confidence_mask)
        #image, character_bboxes, words, confidence_mask, confidences = self.load_image_gt_and_confidencemask(index)
        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()
        region_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_bboxes = []

        if len(character_bboxes) > 0:
            #print("!!!!Hello@@@")
            region_scores = self.gaussianTransformer.generate_region(region_scores.shape, character_bboxes)
            
           
            #print(region_scores)
            affinity_scores, affinity_bboxes = \
                self.gaussianTransformer.generate_affinity(region_scores.shape, character_bboxes, words)
            #print(affinity_scores)
            # from myutils.cv_utils import show_img_mask
            # show_img_mask(image[:, :, ::-1], region_scores / 255., save_name="tmp7.jpg")
            # show_img_mask(image[:, :, ::-1], affinity_scores / 255., save_name="tmp8.jpg")
        if self.viz:
            self.saveImage(self.get_imagename(index), image.copy(), character_bboxes, affinity_bboxes, region_scores,
                           affinity_scores,
                           confidence_mask)
            
        random_transforms = [image, region_scores, affinity_scores, confidence_mask]
        random_transforms = random_crop(random_transforms, (self.target_size, self.target_size), character_bboxes)
        random_transforms = random_horizontal_flip(random_transforms)
        random_transforms = random_rotate(random_transforms)

        cvimage, region_scores, affinity_scores, confidence_mask = random_transforms

        region_scores = self.resizeGt(region_scores)
        affinity_scores = self.resizeGt(affinity_scores)
        confidence_mask = self.resizeGt(confidence_mask)

        if self.viz:
            self.saveInput(self.get_imagename(index), cvimage, region_scores, affinity_scores, confidence_mask)
        image = Image.fromarray(cvimage)
        image = image.convert('RGB')
        image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)

        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))
        image = torch.from_numpy(image).float().permute(2, 0, 1)
        region_scores_torch = torch.from_numpy(region_scores / 255).float()
        affinity_scores_torch = torch.from_numpy(affinity_scores / 255).float()
        confidence_mask_torch = torch.from_numpy(confidence_mask).float()
        return image, region_scores_torch, affinity_scores_torch, confidence_mask_torch, confidences
        
        # random_transforms = [image, region_scores, affinity_scores, confidence_mask]
        # # randomcrop = eastrandomcropdata((768,768))
        # #region_image, affinity_image, character_bboxes = randomcrop(region_image, affinity_image, character_bboxes)
        # # print(random_transforms)
        # random_transforms = random_crop(random_transforms, (self.target_size, self.target_size), character_bboxes)
        # #print(random_transforms)
        # image, region_image, affinity_image, confidence_mask = random_transforms
        # image = Image.fromarray(image)
        # image = image.convert('RGB')
        # # image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)

        # image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
        #                                       variance=(0.229, 0.224, 0.225))
        # image = image.transpose(2, 0, 1)

        # #resize label
        # region_image = self.resizeGt(region_image)
        # affinity_image = self.resizeGt(affinity_image)
        # confidence_mask = self.resizeGt(confidence_mask)

        # region_image = region_image.astype(np.uint8) / 255
        # affinity_image = affinity_image.astype(np.uint8) / 255
        # confidence_mask = confidence_mask.astype(np.uint8)
        
        # return image, region_image, affinity_image, confidence_mask, confidences

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        return self.pull_item(index)

if __name__ == "__main__":
    #data_dir_list = {"synthtext":"/media/neeraj/SSD/DS/ST/SynthText"}
    data_dir_list = {"synthtext":"/home/neeraj/Desktop/IFT6759/CRAFT/DS"}
    craft_data = SynthTextDataLoader(768, data_dir_list)
    for index in range(10000):
        image, character_bboxes, words, confidence_mask, confidences, img_path = craft_data.load_synthtext_image_gt(index)
        # # 测试
        # image = cv2.imread("/media/yanhai/disk21/SynthTextData/SynthText/8/ballet_106_0.jpg")
        # character_bboxes = np.array([[[[423.16126397,  22.26958901],
        #                              [438.2997574,   22.46075572],
        #                              [435.54895424,  40.15739982],
        #                              [420.17946701,  39.82138755]],
        #                             [[439.60847343,  21.60559248],
        #                              [452.61288403,  21.76391911],
        #                              [449.95797159,  40.47241401],
        #                              [436.74150236,  40.18347166]],
        #                             [[450.66887979,  27.0241972 ],
        #                              [466.31976402,  27.25747678],
        #                              [464.5848793,   40.79219178],
        #                              [448.74896556,  40.44598236]],
        #                             [[466.31976402,  27.25747678],
        #                              [482.22585715,  27.49456029],
        #                              [480.68235876,  41.14411963],
        #                              [464.5848793,   40.79219178]],
        #                             [[479.76190495,  27.45783459],
        #                              [498.3934528,   27.73554156],
        #                              [497.04793842,  41.50190876],
        #                              [478.18853922,  41.08959901]],
        #                             [[504.59927448,  28.73896576],
        #                              [512.20555863,  28.85582217],
        #                              [511.1101386,   41.80934074],
        #                              [503.4152019,   41.64111176]]]])
        # character_bboxes = character_bboxes.astype(np.int)
        gaussian_map = np.zeros(image.shape, dtype=np.uint8)
        gen = GaussianTransformer(200, 1.5)
        gen.gen_circle_mask()

        region_image = gen.generate_region(image.shape, character_bboxes)
        affinity_image, affinities = gen.generate_affinity(image.shape, character_bboxes, words)

        random_transforms = [image, region_image, affinity_image, confidence_mask]
        # randomCrop = EastRandomCropData((768,768))
        # region_image, affinity_image, character_bboxes = randomCrop(region_image, affinity_image, character_bboxes)


        random_transforms = random_crop(random_transforms, (768, 768), character_bboxes)
        image, region_image, affinity_image, confidence_mask = random_transforms
        region_image = cv2.applyColorMap(region_image, cv2.COLORMAP_JET)
        affinity_image = cv2.applyColorMap(affinity_image, cv2.COLORMAP_JET)

        region_image = cv2.addWeighted(region_image, 0.3, image, 1.0, 0)
        affinity_image = cv2.addWeighted(affinity_image, 0.3, image, 1.0, 0)

        # cv2.imshow("gaussion", gaussion)
        # cv2.waitKey(0)
        for boxes in character_bboxes:
            for box in boxes:
                # image = cv2.imread(img_path)
                enlarge = enlargebox(box.astype(np.int), image.shape[0], image.shape[1])
                # print("enlarge:", enlarge)
                # gaussion = gen.generate_region(image.shape, np.array([[box]]))
                # gaussion = cv2.applyColorMap(gaussion, cv2.COLORMAP_JET)
        #         cv2.polylines(image, [enlarge], True, (0, 0, 255), 1)
        #         cv2.polylines(image, [box.astype(np.int)], True, (0, 255, 255), 1)
        #         cv2.polylines(region_image, [enlarge], True, (0, 0, 255), 1)
        #         cv2.polylines(region_image, [box.astype(np.int)], True, (0, 255, 255), 1)
        #         cv2.polylines(affinity_image, [box.astype(np.int)], True, (0, 255, 255), 1)
        # for box in affinities:
        #     cv2.polylines(affinity_image, [box.astype(np.int)], True, (0, 0, 255), 1)
        stack_image = np.hstack((region_image, affinity_image))
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", stack_image)
        cv2.waitKey(0)






