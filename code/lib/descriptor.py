# feature descriptor using SIFT
import numpy as np
import cv2

class SIFT:
    def __init__(self):
        self.subpatch_size = 16
        # params for gaussian kernal
        self.ksize = 3
        self.sigma = 5

    def gray_gaussianBlurr(self, img):
        # print('[SIFT] Convert img to grayscale and apply gaussian filter')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray_img, (self.ksize, self.ksize), self.sigma)
    
    def detect(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(gray, None)
        self.kps.append(kp)
        self.descs.append(des)
        return kp, des

    def compute_subpatch_descriptor(self, m, theta, bins = 8):
        # Assign theta to bins
        theta = theta // (360. / bins)
        
        # histogram : 8 orientations x 4x4 histogram array = 128 dimensions    
        B = 4  # radius of one of the 4x4
        histogram = np.empty((0, 128)).astype('float64')
        for i in range(0, 16, B):
            for j in range(0, 16, B):
                hist_subpatch = np.zeros(8)
                weight = m[i:i+B, j:j+B].ravel()
                bins = theta[i:i+B, j:j+B].ravel().astype('uint8')
                np.add.at(hist_subpatch, bins, weight)
                histogram = np.append(histogram, hist_subpatch)

        histogram_norm = histogram / np.sum(histogram) # normalize  
        histogram_norm = np.clip(histogram_norm, a_min=None, a_max=0.2)
        histogram_norm /= np.sum(histogram_norm) # renormalize

        return histogram_norm

    def compute_descriptor(self, img, kps):
        h, w = img.shape[:2]
        img = self.gray_gaussianBlurr(img) if len(img.shape) == 3 else img
        r = self.subpatch_size // 2 

        # m : image gradients (magnitude)
        Iy, Ix = np.gradient(img)
        m = np.sqrt(np.sum((Ix ** 2, Iy ** 2), axis=0))
        m = cv2.GaussianBlur(m, (7, 7), 0.5)

        # theta : gradient vector orientation
        theta = np.rad2deg(np.arctan2(Iy, Ix)) % 360

        kps_of_descs, descs= [], []
        for kp in kps:
            row, col = kp

            # Boundary check
            if row-r < 0 or row+r >= h or col-r < 0 or col+r >= w:
                continue

            # Get keypoint orientation and subpatch
            theta_kp = theta[row, col]
            m_subpatch = m[row-r:row+r, col-r:col+r]
        
            # Get relative orientation
            relative_theta = theta[row-r:row+r, col-r:col+r] - theta_kp
            relative_theta[relative_theta < 0] += 360
            
            # Get descriptor
            desc = self.compute_subpatch_descriptor(m_subpatch, relative_theta)

            kps_of_descs.append((row, col))
            descs.append(desc)

        return kps_of_descs, descs