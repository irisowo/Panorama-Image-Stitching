import os
import cv2
import time
import argparse
import numpy as np
import multiprocessing as mp

import lib.utils as utils
import lib.detector as detector
import lib.descriptor as descriptor
import lib.matcher as matcher
import lib.aligner as aligner
import lib.blender as blender 


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type = str, default = '../data/home', help = 'Input directory')
    parser.add_argument('--e2e', action = 'store_true', help = 'If the images are taken end-to-end')
    parser.add_argument('--cache', action = 'store_true', help = 'Read cylindrical projected images from cache')

    args = parser.parse_args()
    return args


class Image_Stitch_Solution():
    def __init__(self, args=None) -> None:
        self.args = args
        self.detector = detector.Harris_Corner()
        self.descriptor = descriptor.SIFT()
        self.matcher = matcher.Brute_Force_Matcher()
        self.aligner = aligner.RANSAC_aliner()
        self.blender = blender.Linear_Blender()

    def solve(self, input_imgs):
        imgs = input_imgs.copy()

        print(f'[Feature] Detecting Feature')
        detect_kps_list = []
        detect_imgs_list = []
        for img in imgs:
            # 1. Harris Corner Detector for keypoints
            keypoints, gray_blur_ig = self.detector.detect(img)
            detect_kps_list.append(keypoints)
            detect_imgs_list.append(gray_blur_ig)

        print(f'[Feature] Applying Descriptor')
        kps_list = []
        descs_list = []
        for keypoints, gray_blur_img in zip(detect_kps_list, detect_imgs_list):
            # 2. SIFT descriptors
            kp, des = self.descriptor.compute_descriptor(gray_blur_img, keypoints)
            # kp, des = sift.compute_descriptor(img, keypoints)
            kps_list.append(kp)
            descs_list.append(des)
        
        print(f'[Feature Matching] Matching Descriptors')
        matches_imgs = []
        for i, img in enumerate(imgs[:-1]):
            matches = self.matcher.match(descs_list[i], descs_list[i+1])
            matches_imgs.append(matches)
            # utils.plot_matching(img, imgs[i+1], kps_list[i], kps_list[i+1], good_matches)
    
        print(f'[Pair-Wise Alignment] Computing Shifts Between Images')
        shifts_x, shifts_y = [0], [0]
        for i, matches in enumerate(matches_imgs):

            # retrieve the matched keypoints
            match_kp1s = np.array(kps_list[i])[matches[:, 0]]
            match_kp2s = np.array(kps_list[i+1])[matches[:, 1]]

            # RANSAC alignment
            shift_x, shift_y = self.aligner.align(match_kp1s, match_kp2s)
            shifts_x.append(shift_x)
            shifts_y.append(shift_y)

        # deal with the drifting probelm (end-to-end alignment)
        shifts_y_accum = np.cumsum(shifts_y)
        drift_y = (shifts_y_accum[-1] + shifts_y[-1]) / (len(imgs) - 1) if self.args.e2e else (shifts_y_accum[-1]) / (len(imgs) - 1)
            
        print("[Blending] Stitching Images")
        stitch_img = imgs[0]
        for i in range(1, len(imgs)):
            shift_x = shifts_x[i]
            shift_y = round(shifts_y_accum[i] - drift_y * i)
            stitch_img = self.blender.blend(imgs[i], stitch_img, shift_x, shift_y)

        # stitch the last image to the first image to check the end-to-end alignment
        if self.args.e2e: 
            stitch_img = self.blender.blend(imgs[0], stitch_img, shifts_x[-1], 0.0)

        cv2.imshow('Panorama', stitch_img)
        cv2.waitKey(0)
        return stitch_img


if __name__ == '__main__':
    args = get_args()
    dataset = args.indir.split('/')[-1]

    imgs = []

    print('[Preprocess] Cylindrical Projecting')
    if args.cache:
        # read from cache directory (start with cy_)
        n = len([name for name in os.listdir(f'../data/cy_{dataset}') if name.startswith('cylindrical_')])
        imgs = [cv2.imread(f'../data/cy_{dataset}/cylindrical_{i}.png') for i in range(n)]
    else:
        images, focal_lengths = utils.read_images_and_focal_lengths(args.indir, 0)
        # check path
        if not os.path.exists(f'../data/cy_{dataset}'):
            os.makedirs(f'../data/cy_{dataset}')
        # cylindrical projection
        with mp.Pool(mp.cpu_count()) as pool:
            imgs = pool.starmap(utils.cylindrical_projection, [(img, f) for img, f in zip(images, focal_lengths)])
            for i, img in enumerate(imgs):
                cv2.imwrite(f'../data/cy_{dataset}/cylindrical_{i}.png', img)

    solution = Image_Stitch_Solution(args)
    solution.solve(imgs)