from math import exp
import numpy as np
import cv2
import matplotlib as plt
import os , math , sys
import traceback
sys.path.append('/home/neeraj/Desktop/IFT6759/CRAFT')
from gaussianMask import imgprocess

from data.boxEnlarge import enlargebox

class GaussianTransformer(object):

    #def __init__(self, imgSize=200, enlargeSize =1.5):
    def __init__(self, imgSize=512, region_threshold=0.4,
                 affinity_threshold=0.2):
        self.imgSize = imgSize
        isotropicGrayScaleImage, isotropicGrayscaleImageColor = self.gen_gaussian_heatmap()
        self.standardGaussianHeat = isotropicGrayScaleImage
       # self.enlargeSize = enlargeSize
        _, binary = cv2.threshold(self.standardGaussianHeat, region_threshold * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.regionbox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        # print("regionbox", self.regionbox)
        _, binary = cv2.threshold(self.standardGaussianHeat, affinity_threshold * 255, 255, 0)
        np_contours = np.roll(np.array(np.where(binary != 0)), 1, axis=0).transpose().reshape(-1, 2)
        x, y, w, h = cv2.boundingRect(np_contours)
        self.affinitybox = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)
        # print("affinitybox", self.affinitybox)
        self.oribox = np.array([[0, 0, 1], [imgSize - 1, 0, 1], [imgSize - 1, imgSize - 1, 1], [0, imgSize - 1, 1]],
                               dtype=np.int32)

    def gen_gaussian_heatmap(self):
        circle_mask = self.gen_circle_mask()
        imgSize = self.imgSize
        isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)
        imgSize = self.imgSize
        isotropicGrayscaleImage = np.zeros((imgSize, imgSize), np.float32)

        for i in range(imgSize):
            for j in range(imgSize):
                isotropicGrayscaleImage[i, j] = 1 / 2 / np.pi / (40 ** 2) * np.exp(
                    -1 / 2 * ((i - imgSize / 2) ** 2 / (40 ** 2) + (j - imgSize / 2) ** 2 / (40 ** 2)))
       
        isotropicGrayscaleImage = isotropicGrayscaleImage * circle_mask
        isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage)).astype(np.float32)

        isotropicGrayscaleImage = (isotropicGrayscaleImage / np.max(isotropicGrayscaleImage) * 255).astype(np.uint8)
        isotropicGrayscaleImageColor = cv2.applyColorMap(isotropicGrayscaleImage, cv2.COLORMAP_JET)
        return isotropicGrayscaleImage, isotropicGrayscaleImageColor

   
    def gen_circle_mask(self):
        imgSize = self.imgSize
        circle_img = np.zeros((imgSize, imgSize), np.uint8)
        circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 1, -1)

        # circle_mask = cv2.circle(circle_img, (imgSize//2, imgSize//2), imgSize//2, 255, -1)
        # circle_mask = cv2.applyColorMap(circle_mask, cv2.COLORMAP_JET)
        # cv2.imshow("circle", circle_mask)
        # cv2.waitKey(0)
        return circle_mask

    
    def enlargeBox(self, box, imgh, imgw):
        boxw = box[1][0] - box[0][0]
        boxh = box[2][1] - box[1][1]

        if imgh <= boxh or imgw <= boxw:
            return box

        enlargew = boxw * 0.5
        enlargeh = boxh * 0.5

        box[0][0], box[0][1] = max(0, box[0][0] - int(enlargew*0.5)), max(0, box[0][1] - int(enlargeh*0.5))
        box[1][0], box[1][1] = min(imgw, box[1][0] + int(enlargew*0.5)), max(0, box[1][1] - int(enlargeh*0.5))
        box[2][0], box[2][1] = min(imgw, box[2][0] + int(enlargew*0.5)), min(imgh, box[2][1] + int(enlargeh*0.5))
        box[3][0], box[3][1] = max(0, box[3][0] - int(enlargew*0.5)), min(imgh, box[3][1] + int(enlargeh*0.5))

        return box

    def four_point_transform(self, target_bbox, save_dir=None):
        '''
        :param target_bbox:bbox
        :param save_dir:None，save_dir
        :return:
        '''
        width, height = np.max(target_bbox[:, 0]).astype(np.int32), np.max(target_bbox[:, 1]).astype(np.int32)
        right = self.standardGaussianHeat.shape[1] - 1
        bottom = self.standardGaussianHeat.shape[0] - 1
        ori = np.array([[0, 0], [right, 0],
                        [right, bottom],
                        [0, bottom]], dtype="float32")
        M = cv2.getPerspectiveTransform(ori, target_bbox)
        warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (int(width), int(height)))
        warped = np.array(warped, np.uint8)
        if save_dir:
            warped_color = cv2.applyColorMap(warped, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_dir, 'warped.jpg'), warped_color)

        return warped, width, height

    # def add_character(self, image, bbox, singal = None):
    #     # bbox = self.enlargeBox(bbox, image.shape[0], image.shape[1])
    #     print(image)
    #     bbxo_copy = bbox.copy()
    #     bbox = enlargebox(bbox, image.shape[0], image.shape[1])
    #     if singal == "affinity":
    #         bbox[0][0], bbox[1][0], bbox[2][0], bbox[3][0] = bbxo_copy[0][0], bbxo_copy[1][0], bbxo_copy[2][0], bbxo_copy[3][0]

        
    #     if np.any(bbox < 0) or np.any(bbox[:, 0] > image.shape[1]) or np.any(bbox[:, 1] > image.shape[0]):
    #         return image
    #     ori_box = bbox
    #     top_left = np.array([np.min(bbox[:, 0]), np.min(bbox[:, 1])]).astype(np.int32)
    #     point = top_left
    #     bbox -= top_left[None, :]
    #     transformed, width, height = self.four_point_transform(bbox.astype(np.float32))
    #     # if width * height < 10:
    #     #     return image
    #     try:
    #         score_map = image[top_left[1]:top_left[1] + transformed.shape[0],
    #                     top_left[0]:top_left[0] + transformed.shape[1]]
    #         score_map = np.where(transformed > score_map, transformed, score_map)
    #         image[top_left[1]:top_left[1] + transformed.shape[0],
    #         top_left[0]:top_left[0] + transformed.shape[1]] = score_map
    #     except Exception as e:
    #         # print('tansformed shape:{}\n image top_left shape:{}\n top transformed shape:{}\n width and hright:{}\n ori box:{}\n top left:{}\n point:{}\n min width height:{}\n bbox:{}\n'
    #         #       .format(transformed.shape, image[top_left[1]:top_left[1],
    #         # top_left[0]:top_left[0]].shape, image[top_left[1]:top_left[1] + transformed.shape[0],
    #         # top_left[0]:top_left[0] + transformed.shape[1]].shape, (width, height), ori_box, top_left,point,
    #         #       np.array([np.min(ori_box[:, 0]), np.min(ori_box[:, 1])]).astype(np.int32),
    #         #       ori_box-np.array([np.min(ori_box[:, 0]), np.min(ori_box[:, 1])]).astype(np.int32)))
    #         print('second filter {} {} {}'.format(width,height,singal))
    #     return image


    def generate_region(self, image_size, bboxes):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        for i in range(len(bboxes)):
            character_bbox = np.array(bboxes[i].copy())
            for j in range(bboxes[i].shape[0]):
                
                target1 = self.add_region_character(target, character_bbox[j])
                
                # try:
                #      target1
                # except Exception as e:
                #     print(e)
                #     print(traceback.format_exc())

            # print('::: Hello::::')
            # print(str(height) +':'+ str(width))
        return target1

    
    def add_region_character(self, image, target_bbox, regionbox=None):

        if np.any(target_bbox < 0) or np.any(target_bbox[:, 0] > image.shape[1]) or np.any(
                target_bbox[:, 1] > image.shape[0]):
            return image
        affi = False
        if regionbox is None:
            regionbox = self.regionbox.copy()
        else:
            affi = True

        M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(target_bbox))
        oribox = np.array(
            [[[0, 0], [self.imgSize - 1, 0], [self.imgSize - 1, self.imgSize - 1], [0, self.imgSize - 1]]],
            dtype=np.float32)
        test1 = cv2.perspectiveTransform(np.array([regionbox], np.float32), M)[0]
        real_target_box = cv2.perspectiveTransform(oribox, M)[0]
        # print("test\ntarget_bbox", target_bbox, "\ntest1", test1, "\nreal_target_box", real_target_box)
        real_target_box = np.int32(real_target_box)
        real_target_box[:, 0] = np.clip(real_target_box[:, 0], 0, image.shape[1])
        real_target_box[:, 1] = np.clip(real_target_box[:, 1], 0, image.shape[0])

        # warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
        # warped = np.array(warped, np.uint8)
        # image = np.where(warped > image, warped, image)
        if np.any(target_bbox[0] < real_target_box[0]) or (
                target_bbox[3, 0] < real_target_box[3, 0] or target_bbox[3, 1] > real_target_box[3, 1]) or (
                target_bbox[1, 0] > real_target_box[1, 0] or target_bbox[1, 1] < real_target_box[1, 1]) or (
                target_bbox[2, 0] > real_target_box[2, 0] or target_bbox[2, 1] > real_target_box[2, 1]):
            # if False:
            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), M, (image.shape[1], image.shape[0]))
            warped = np.array(warped, np.uint8)
            image = np.where(warped > image, warped, image)
            #print(image)
            #print("warped@@")
            # _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
            # warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
            # warped = np.array(warped, np.uint8)
            #
            # if affi:
            #  print("warped", warped.shape, real_target_box, target_bbox, _target_box)
            # # cv2.imshow("1123", warped)
            # # cv2.waitKey()
            # image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
            #                                        image[ymin:ymax, xmin:xmax])
        else:
            xmin = real_target_box[:, 0].min()
            xmax = real_target_box[:, 0].max()
            ymin = real_target_box[:, 1].min()
            ymax = real_target_box[:, 1].max()

            width = xmax - xmin
            height = ymax - ymin
            _target_box = target_bbox.copy()
            _target_box[:, 0] -= xmin
            _target_box[:, 1] -= ymin
            _M = cv2.getPerspectiveTransform(np.float32(regionbox), np.float32(_target_box))
            warped = cv2.warpPerspective(self.standardGaussianHeat.copy(), _M, (width, height))
            warped = np.array(warped, np.uint8)
            if warped.shape[0] != (ymax - ymin) or warped.shape[1] != (xmax - xmin):
                print("region (%d:%d,%d:%d) warped shape (%d,%d)" % (
                    ymin, ymax, xmin, xmax, warped.shape[1], warped.shape[0]))
                return image
            # if affi:
            #     print("warped", warped.shape, real_target_box, target_bbox, _target_box)
            # cv2.imshow("1123", warped)
            # cv2.waitKey()
            #print("warped!!!!")
            image[ymin:ymax, xmin:xmax] = np.where(warped > image[ymin:ymax, xmin:xmax], warped,
                                                   image[ymin:ymax, xmin:xmax])
            #print(image)    
        return image

    # def add_affinity(self, image, bbox_1, bbox_2):
    #     center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
    #     tl = (bbox_1[0:2].sum(0) + center_1) / 3
    #     bl = (bbox_2[0:2].sum(0) + center_2) / 3
    #     tr = (bbox_2[2:4].sum(0) + center_2) / 3
    #     br = (bbox_1[2:4].sum(0) + center_1) / 3

    #     affinity = np.array([tl, bl, tr, br]).astype(np.uint8)
    
    #     return self.add_region_character(image, affinity.copy(), singal='affinity'), np.expand_dims(affinity, axis=0)
    def add_affinity(self, image, bbox_1, bbox_2):
        center_1, center_2 = np.mean(bbox_1, axis=0), np.mean(bbox_2, axis=0)
        tl = np.mean([bbox_1[0], bbox_1[1], center_1], axis=0)
        bl = np.mean([bbox_1[2], bbox_1[3], center_1], axis=0)
        tr = np.mean([bbox_2[0], bbox_2[1], center_2], axis=0)
        br = np.mean([bbox_2[2], bbox_2[3], center_2], axis=0)

        affinity = np.array([tl, tr, br, bl])

        return self.add_affinity_character(image, affinity.copy()), np.expand_dims(affinity, axis=0)
    
    def add_affinity_character(self, image, target_bbox):
        return self.add_region_character(image, target_bbox, self.affinitybox)

    # def generate_region(self, image_size, bboxes):
    #     height, width, channel = image_size
    #     target = np.zeros([height, width], dtype=np.uint8)
        
    #     for i in range(len(bboxes)):
    #         character_bbox = np.array(bboxes[i])
    #        # print(len(bboxes))enlargebox
    #         for j in range(bboxes[i].shape[0]):
    #             print(j)
    #             target = self.add_region_character(target, character_bbox[j])
               
        #return target
 
    def saveGaussianHeat(self):
        images_folder = os.path.abspath(os.path.dirname(__file__)) + '/images'
        cv2.imwrite(os.path.join(images_folder, 'standard.jpg'), self.standardGaussianHeat)
        warped_color = cv2.applyColorMap(self.standardGaussianHeat, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(images_folder, 'standard_color.jpg'), warped_color)
        standardGaussianHeat1 = self.standardGaussianHeat.copy()
        standardGaussianHeat1[standardGaussianHeat1 < (0.4 * 255)] = 255
        threshhold_guassian = cv2.applyColorMap(standardGaussianHeat1, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(images_folder, 'threshhold_guassian.jpg'), threshhold_guassian)

    def generate_affinity(self, image_size, bboxes, words):
        height, width = image_size[0], image_size[1]
        target = np.zeros([height, width], dtype=np.uint8)
        affinities = []
        for i in range(len(words)):
            character_bbox = np.array(bboxes[i])
            total_letters = 0
            for char_num in range(character_bbox.shape[0] - 1):
                target, affinity = self.add_affinity(target, character_bbox[total_letters],
                                                     character_bbox[total_letters + 1])
                affinities.append(affinity)
                total_letters += 1
        if len(affinities) > 0:
            affinities = np.concatenate(affinities, axis=0)
        return target, affinities


if __name__ == '__main__':
    # gaussian = GaussianTransformer(200, 1.5)
    # gaussian.saveGaussianHeat()
    image = np.zeros((500,500,3),dtype=np.uint8)
    cv2.imwrite ('img.png' , image)
    gen = GaussianTransformer(200, 1.5)
    cm = gen.gen_circle_mask()
    #print(gen)
    bbox = np.array([[[60, 140], [110, 160], [110, 260], [60, 230]], [[110, 165], [180, 165], [180, 255], [110, 255]]])
    bbox = bbox[np.newaxis, :, :,:]
    region_image = gen.generate_region(image.shape, bbox)
    #print(type(region_image)
    region_image = cv2.applyColorMap(region_image, cv2.COLORMAP_JET)
    
    affinity_image, affinities = gen.generate_affinity(image.shape, bbox, [[1,2]])
    affinity_image = cv2.applyColorMap(affinity_image, cv2.COLORMAP_JET)
    target_bbox = np.array([[45, 135], [135, 135], [135, 295], [45, 295]], dtype=np.int8)

    for boxes in bbox:
        for box in boxes:
            # cv2.rectangle(image, tuple(box[0]), tuple(box[2]), (0, 255, 255), 2)
            enlarge = enlargebox(box, image.shape[1], image.shape[0])
            cv2.polylines(region_image, [box], True, (0, 255, 255), 2)
            cv2.polylines(region_image, [enlarge], True, (0, 0, 255), 2)
            cv2.polylines(affinity_image, [box], True, (0, 255, 255), 2)
            # cv2.polylines(affinity_image, [enlarge], True, (0, 0, 255), 2)

    cv2.polylines(affinity_image, [affinities[0].astype(int)], True, (255, 0, 255), 2)

    cv2.imshow('test', np.hstack((region_image, affinity_image)))
    cv2.waitKey(0)



