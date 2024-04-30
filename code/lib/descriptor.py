# feature descriptor using SIFT
import numpy as np
import cv2
import math
import pywt


class DESCRIPTOR():
    def __init__(self):
        # Do gaussian blur on the image in the descriptor stage
        self.ksize_for_desc = 7
        self.sigma_for_desc = 0.5
        # Do gaussian blur on the image in the detector stage
        #(The params are only used if the img is not grayscale)
        self.ksize = 3
        self.sigma = 5

    def gray_gaussianBlurr(self, img):
        # print('[SIFT] Convert img to grayscale and apply gaussian filter')
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray_img, (self.ksize, self.ksize), self.sigma)

    def bounday_check(self, row, col, h, w, r):
        return row-r < 0 or row+r >= h or col-r < 0 or col+r >= w


class SIFT(DESCRIPTOR):
    def __call__(self):
        print(". . . Using SIFT Descriptor . . .")

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
        r = 16 // 2 

        # m : image gradients (magnitude)
        Iy, Ix = np.gradient(img)
        m = np.sqrt(np.sum((Ix ** 2, Iy ** 2), axis=0))
        m = cv2.GaussianBlur(m, (self.ksize_for_desc, self.ksize_for_desc), self.sigma_for_desc)

        # theta : gradient vector orientation
        theta = np.rad2deg(np.arctan2(Iy, Ix)) % 360

        kps_of_descs, descs= [], []
        for kp in kps:
            row, col = kp

            # Boundary check
            if self.bounday_check(row, col, h, w, r):
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
    
    def detect(self, img):
        pass


class MOPS(DESCRIPTOR):
    def __call__(self):
        print(". . . Using MOPS Descriptor . . .")

    def comput_orientation_image(self, img):
        Iy, Ix = np.gradient(img)
        # orientationImg = np.zeros(img.shape[:2])
        # return np.degrees(np.arctan2(Iy.flatten(), Ix.flatten()).reshape(orientationImg.shape))
        return (np.arctan2(Iy, Ix)) % (2 * np.pi)

    def compute_descriptor(self, img, kps):
        h, w = img.shape[:2]
        img = self.gray_gaussianBlurr(img) if len(img.shape) == 3 else img
        img = cv2.GaussianBlur(img, (self.ksize_for_desc, self.ksize_for_desc), self.sigma_for_desc) # same as sift
        orientationImg = self.comput_orientation_image(img)

        r = (40 * math.sqrt(2)) // 2 # times sqrt(2) to avoid overflow after rotation
        window_size = 8

        kps_of_descs, descs= [], []
        for kp in kps:
            row, col = kp
            if self.bounday_check(row, col, h, w, r):
                continue
            rad = orientationImg[row][col]
            rotM = np.array([[math.cos(rad), math.sin(rad), 0],
                             [-math.sin(rad), math.cos(rad), 0],
                             [0, 0, 1]])
            scaleM = np.diag([0.2, 0.2, 1])
            transM = np.array([[1, 0, -col], [0, 1, -row], [0, 0, 1]])
            transM2 = np.array([[1, 0 , window_size/2], [0, 1, window_size/2]])
            affineM = transM2 @ scaleM @ rotM @ transM
            # Call the warp affine function to do the mapping which expects a 2x3 matrix
            feature = cv2.warpAffine(img, affineM[:2],
                (window_size, window_size), flags=cv2.INTER_LINEAR)

            # Intensity normalization
            feature = ((feature - np.mean(feature)) / (np.std(feature) + 1e-6))

            # Haar wavelet transform
            # coeffs = pywt.dwt2(img, 'haar')
            # feature = pywt.idwt2(coeffs, 'haar')

            # Append to the list
            kps_of_descs.append(kp)
            descs.append(feature.ravel())
        return kps_of_descs, descs