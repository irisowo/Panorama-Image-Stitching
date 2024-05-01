import os
import cv2
import argparse
import numpy as np

import lib.utils as utils
import lib.cylinder as cylinder
import lib.detector as detector
import lib.descriptor as descriptor
import lib.matcher as matcher
import lib.aligner as aligner
import lib.blender as blender 


def get_args():
    parser = argparse.ArgumentParser(description="Image stitching parameters")
    parser.add_argument('--indir', type = str, default = '../data/home', help = "Input directory")
    parser.add_argument('--desc', type=str, default='sift', choices=['sift', 'mops'], help="Descriptor type")
    parser.add_argument('--blend', type=float, default = 0.1, help = "Blending ratio")
    parser.add_argument('--e2e', action = 'store_true', help = "If the images are taken end-to-end")
    parser.add_argument('--save', action = 'store_true', help = "Save the stitched image. If not, show the image")
    parser.add_argument('--cache', action = 'store_true', help = "Read cylindrical projected images from cache")

    args = parser.parse_args()
    return args


class Image_Stitch_Solution():
    def __init__(self, args=None) -> None:
        self.args = args
        self.detector = detector.Harris_Corner()
        self.descriptor = descriptor.MOPS() if(args.desc == 'mops') else descriptor.SIFT()
        self.matcher = matcher.Brute_Force_Matcher()
        self.aligner = aligner.RANSAC_Aliner()
        self.blender = blender.Linear_Blender(args.blend)

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

        print(f'[Feature] Applying {self.descriptor()} Descriptor')
        kps_list = []
        descs_list = []
        for keypoints, gray_blur_img in zip(detect_kps_list, detect_imgs_list):
            kp, des = self.descriptor.compute_descriptor(gray_blur_img, keypoints)
            # kp, des = self.descriptor.compute_descriptor(img, keypoints)
            kps_list.append(kp)
            descs_list.append(des)
        
        print(f'[Feature Matching] Matching Descriptors')
        matches_imgs = []
        for i, img in enumerate(imgs[:-1]):
            matches = self.matcher.match(descs_list[i], descs_list[i+1])
            matches_imgs.append(matches)
            # utils.plot_matching(img, imgs[i+1], kps_list[i], kps_list[i+1], matches)
    
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
        drift_y = (shifts_y_accum[-1]) / (len(imgs) - 1) # if args.e2e else 0.0
        
        print("[Blending] Stitching Images")
        stitch_img = imgs[0]
        for i in range(1, len(imgs)):
            shift_x = shifts_x[i]
            shift_y = round(shifts_y_accum[i] - drift_y * i)
            stitch_img = self.blender.blend(imgs[i], stitch_img, shift_x, shift_y)
            # stitch_img = self.blender.blend(imgs[i], stitch_img, shift_x, 0)

        # stitch the last image to the first image to check the end-to-end alignment
        if self.args.e2e: 
            stitch_img = self.blender.blend(imgs[0], stitch_img, shifts_x[-1], 0.0)
        # stitch_img = utils.crop(stitch_img, 100, stitch_img.shape[0] - 75)
        stitch_img = utils.crop(stitch_img, 40, stitch_img.shape[0] - 40)
    
        if(args.save):
            cv2.imwrite('../panorama.png', stitch_img)
        else:
            cv2.imshow('Panorama', stitch_img)
            cv2.waitKey(0)
        return stitch_img


if __name__ == '__main__':
    args = get_args()

    dataset = args.indir.split('/')[-1]
    cache_dir = f'../data/cy_{dataset}'
    utils.check_and_mkdir(cache_dir)

    print('[Preprocess] Cylindrical Projecting')
    imgs = []
    if args.cache: # read from cache directory
        n = len([name for name in os.listdir(cache_dir) if name.startswith('cylindrical_')])
        imgs = [cv2.imread(os.path.join(cache_dir, f'cylindrical_{i}.png')) for i in range(n)]
    else: # read from indr
        images, focal_lengths = utils.read_images_and_focal_lengths(args.indir, 0)
        projecter = cylinder.Cylindrical_Projection()
        imgs = projecter.cylindrical_project_imgs(images, focal_lengths, cache_dir)

    solution = Image_Stitch_Solution(args)
    solution.solve(imgs)

    # utils.do_experiments()
