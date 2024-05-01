import os
import cv2
import numpy as np
import multiprocessing as mp

class Cylindrical_Projection:
    def __init__(self) -> None:
        pass

    def project_img(self, img, f):
        # Let s = f, center of the image = (y0, x0)
        s = f
        h, w, _= img.shape
        y0, x0 = h / 2, w / 2
        cylindrical_img= np.zeros_like(img, dtype=np.uint8)

        def _project_pt(_y, _x):
            dy, dx = _y - y0, _x - x0
            h = dy / (dx ** 2 + f ** 2) ** 0.5
            theta = np.arctan(dx / f)
            return [round(y0 + (s * h)), round(x0 + (s * theta))]

        # Warping
        for y in range(h):
            for x in range(w):
                y_, x_ = _project_pt(y, x)
                cylindrical_img[y_, x_] = img[y, x]
                
        # Boundary
        x_start = _project_pt(h/2, 0)[1]
        x_end = _project_pt(h/2, w-1)[1] + 1
        return cylindrical_img.astype('uint8')[:, x_start:x_end]
    
    def cylindrical_project_imgs(self, images, focal_lengths, indir):
        imgs = None
        with mp.Pool(mp.cpu_count()) as pool:
            imgs = pool.starmap(self.project_img, [(img, f) for img, f in zip(images, focal_lengths)])
            for i, img in enumerate(imgs):
                cv2.imwrite(os.path.join(indir, f'cylindrical_{i}.png'), img)
        return imgs