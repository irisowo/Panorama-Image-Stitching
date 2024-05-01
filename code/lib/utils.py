import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_images_and_focal_lengths(img_dir, startIdx=0):
    df = pd.read_csv(os.path.join(img_dir, 'focal_length.csv'), sep=',')
    images = [cv2.imread(os.path.join(img_dir, f)) for f in df.filename]
    focal_length = df.focal_length.astype(float)

    # change the start image from images[0] to images[startIdx]
    images2 = images[startIdx:] +  images[:startIdx]
    focal_length2 = focal_length[startIdx:].tolist() + focal_length[:startIdx].tolist()
    return images2, focal_length2


def crop(img, y_start, y_end):
    cropped_img = img[y_start: y_end, :]
    return cropped_img


def plot_imgs(imgdir, img_names, figname, 
              scale=5, 
              figsize=(15, 5), 
              wspace=0.0,
              hspace=0.0,
              vertical=False):
              
    imgs = [cv2.imread(os.path.join(imgdir, img_name)) for img_name in img_names]
    titles = [img_name.split('.png')[0].split('_')[1] for img_name in img_names]
    n = len(imgs)

    fig, axes = plt.subplots(n, 1, figsize=figsize) if vertical else plt.subplots(1, n, figsize=figsize)
    fig.subplots_adjust(left=0, right=1.0, top=1.0, bottom=0, wspace=wspace, hspace=hspace)
    for i, (img, title) in enumerate(zip(imgs, titles)):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title, fontsize=3 * scale)
        axes[i].axis('off')
    # save the plot
    plt.savefig(figname)


def plot_matching(img1, img2, kps1, kps2, matches):
    _, img1_w, _ = img1.shape
    img1_img2 = np.hstack((img1, img2))
    for idx1, idx2 in matches:
        (y1, x1), (y2, x2) = kps1[idx1], kps2[idx2]
        p1, p2 = (x1, img1_w + x2), (y1, y2) 
        plt.plot(p1, p2, 'o-', markersize=2.5)
    plt.imshow(cv2.cvtColor(img1_img2, cv2.COLOR_BGR2RGB))
    plt.show()


def do_experiments():
    imgdir = '../data/experiment'
    # read images from imgdir
    prefix = 'match'
    img_names = sorted([f for f in os.listdir(imgdir) if (f.endswith('.png') and f.startswith(prefix))])
    plot_imgs(imgdir, img_names, os.path.join(imgdir, f'result_{prefix}.png'),figsize=(15, 5))

    prefix = 'e2e1'
    img_names = sorted([f for f in os.listdir(imgdir) if (f.endswith('.png') and f.startswith(prefix))])
    plot_imgs(imgdir, img_names, os.path.join(imgdir, f'result_{prefix}.png'), figsize=(15, 7), hspace=0.1, vertical=True)

    prefix = 'blendr1'
    img_names = sorted([f for f in os.listdir(imgdir) if (f.endswith('.png') and f.startswith(prefix))])
    plot_imgs(imgdir, img_names, os.path.join(imgdir, f'result_{prefix}.png'), wspace=0.1)

    prefix = 'blendr2'
    img_names = sorted([f for f in os.listdir(imgdir) if (f.endswith('.png') and f.startswith(prefix))])
    plot_imgs(imgdir, img_names, os.path.join(imgdir, f'result_{prefix}.png'), wspace=0.1)

    prefix = 'align'
    img_names = sorted([f for f in os.listdir(imgdir) if (f.endswith('.png') and f.startswith(prefix))])
    plot_imgs(imgdir, img_names, os.path.join(imgdir, f'result_{prefix}.png'), wspace=0.1)
