import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_images_and_focal_lengths(img_dir, startIdx=0):
    df = pd.read_csv(os.path.join(img_dir, 'focal_length.csv'), sep=',')
    images = [cv2.imread(os.path.join(img_dir, f)) for f in df.filename]
    focal_length = df.focal_length.astype(float)

    # change the start image from images[0] to images[startIdx]
    images2 = images[startIdx:] +  images[:startIdx]
    focal_length2 = focal_length[startIdx:].tolist() + focal_length[:startIdx].tolist()
    print()
    return images2, focal_length2


def plot_matching(img1, img2, kps1, kps2, matches):
    _, img1_w, _ = img1.shape
    img1_img2 = np.hstack((img1, img2))
    for idx1, idx2 in matches:
        (y1, x1), (y2, x2) = kps1[idx1], kps2[idx2]
        p1, p2 = (x1, img1_w + x2), (y1, y2) 
        plt.plot(p1, p2, 'o-', markersize=2.5)
    plt.imshow(cv2.cvtColor(img1_img2, cv2.COLOR_BGR2RGB))
    plt.show()


def cylindrical_projection(img, f):
    # Let s = f, center of the image = (y0, x0)
    s = f
    h, w, _= img.shape
    y0, x0 = h / 2, w / 2
    cylindrical_img= np.zeros_like(img, dtype=np.uint8)

    def _cylindrical_warping_pts(_y, _x):
        dy, dx = _y - y0, _x - x0
        h = dy / (dx ** 2 + f ** 2) ** 0.5
        theta = np.arctan(dx / f)
        return [round(y0 + (s * h)), round(x0 + (s * theta))]

    # Warping
    for y in range(h):
      for x in range(w):
        y_, x_ = _cylindrical_warping_pts(y, x)
        cylindrical_img[y_, x_] = img[y, x]
    
    # Boundary
    x_start = _cylindrical_warping_pts(h/2, 0)[1]
    x_end = _cylindrical_warping_pts(h/2, w-1)[1] + 1
    
    return cylindrical_img.astype('uint8')[:, x_start:x_end]

